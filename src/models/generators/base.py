import os.path
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers, regularizers, optimizers, losses, metrics, callbacks, Model, Sequential

import log_service
import models.operators.layers as ops
from models.operators.op_instantiator import OpInstantiator
from utils.nn_utils import compute_tensor_byte_size


@dataclass
class NetworkBuildInfo:
    '''
    Helper class storing relevant info extracted from the cell specification, used for building the actual neural network model.
    '''
    cell_specification: 'list[tuple]'
    blocks: int
    used_lookbacks: 'set[int]'
    unused_block_outputs: 'list[int]'
    use_skip: bool
    used_cell_indexes: 'list[int]'
    reduction_cell_indexes: 'list[int]'
    need_input_norm_indexes: 'list[int]'


class BaseModelGenerator(ABC):
    '''
    Abstract class used as baseline for model generators concrete implementations.
    This class contains all the shared logic between the models, like how the blocks and cells are built, while
    '''

    # TODO: missing max_lookback to adapt inputs based on the actual lookback. For now only 1 or 2 is supported.
    def __init__(self, cnn_hp: dict, arc_params: dict, training_steps_per_epoch: int, output_classes_count: int, input_shape: 'tuple[int, ...]',
                 data_augmentation_model: Optional[Sequential] = None, preprocessing_model: Optional[Sequential] = None,
                 save_weights: bool = False):
        self._logger = log_service.get_logger(__name__)

        self.concat_only_unused = arc_params['concat_only_unused_blocks']
        self.lookback_reshape = arc_params['lookback_reshape']
        self.motifs = arc_params['motifs']
        self.normal_cells_per_motif = arc_params['normal_cells_per_motif']
        self.total_cells = self.motifs * (self.normal_cells_per_motif + 1) - 1
        self.multi_output = arc_params['multi_output']
        self.residual_cells = arc_params['residual_cells']
        self.se_cell_output = arc_params['se_cell_output']
        self.output_classes_count = output_classes_count
        self.input_shape = input_shape
        # basically, it indicates the application domain of the operators, as int (2D for images, 1D for time series)
        self.op_dims = len(input_shape) - 1

        self.lr = cnn_hp['learning_rate']
        self.filters = cnn_hp['filters']
        self.wr = cnn_hp['weight_reg']
        self.use_adamW = cnn_hp['use_adamW']
        self.l2_weight_reg = regularizers.l2(self.wr) if (self.wr is not None and not self.use_adamW) else None
        self.drop_path_keep_prob = 1.0 - cnn_hp['drop_path_prob']
        self.dropout_prob = cnn_hp['softmax_dropout']  # dropout probability on final softmax
        self.cdr_config = cnn_hp['cosine_decay_restart']  # type: dict

        self.save_weights = save_weights

        # necessary for techniques that scale parameters during training, like cosine decay and scheduled drop path
        self.epochs = cnn_hp['epochs']
        self.training_steps_per_epoch = training_steps_per_epoch

        # if not None, data augmentation will be integrated in the model to be performed directly on the GPU
        self.data_augmentation_model = data_augmentation_model
        # if not None, some preprocessing is performed directly in the model (parametric layers like normalization, so that work also for test split)
        self.preprocessing_model = preprocessing_model

        # compute output shapes of each cell, to enable easier tensor shape management
        self.cell_output_shapes = self._compute_cell_output_shapes()

        # op instantiator takes care of handling the instantiation of Keras layers for building the final architecture
        self.op_instantiator = OpInstantiator(len(input_shape), arc_params['block_join_operator'], weight_reg=self.l2_weight_reg)

        # attributes defined below are manipulated and used during model building.
        # defined in class to avoid having lots of parameter passing in each function.

        # used for layers naming and partition dictionary
        self.cell_index = 0
        self.block_index = 0
        self.prev_cell_index = 0

        # info about the actual cell processed and current model outputs
        # noinspection PyTypeChecker
        self.network_build_info = None  # type: NetworkBuildInfo
        self.output_layers = {}

    def alter_macro_structure(self, m: int, n: int, f: int):
        self.motifs = m
        self.normal_cells_per_motif = n
        self.filters = f

        # recompute properties associated to the macro structure parameters
        self.total_cells = self.motifs * (self.normal_cells_per_motif + 1) - 1
        self.cell_output_shapes = self._compute_cell_output_shapes()

    @abstractmethod
    def _generate_network_info(self, cell_spec: 'list[tuple]', use_stem: bool) -> NetworkBuildInfo:
        '''
        Generate a NetworkBuildInfo instance, which stores metadata useful for generating the model from the given cell specification.

        Args:
            cell_spec: cell specification which must be transformed into a TF model.
            use_stem: if the model uses the "ImageNet stem" or not.

        Returns:
            NetworkBuildInfo instance for the provided cell specification.
        '''
        raise NotImplementedError()

    @abstractmethod
    def _compute_cell_output_shapes(self) -> 'list[list[int, ...]]':
        '''
        Computes the expected output shapes of each cell included in the macro-structure.

        Returns:
            a list of tensor shapes, one for each cell output.
        '''
        raise NotImplementedError()

    def _cur_target_shape(self):
        return self.cell_output_shapes[self.cell_index]

    def _compute_partition_size(self, cell_inputs: 'list[tf.Tensor]'):
        input_tensors_size = 0

        for lb in self.network_build_info.used_lookbacks:
            input_tensors_size += compute_tensor_byte_size(cell_inputs[lb])

        return input_tensors_size

    def _build_cell_util(self, filters: int, inputs: list, partitions_dict: dict, reduction: bool = False):
        '''
        Simple helper function for building a cell and quickly return the inputs for next cell.

        Args:
            filters (int): number of filters
            inputs (list<tf.Tensor>): previous cells output, that can be used as inputs in the current cell
            reduction (bool, optional): Build a reduction cell? Defaults to False.

        Returns:
            (list<tf.Tensor>): Usable inputs for next cell
        '''
        # this check avoids building cells not used in actual final model (cells not linked to output), also removing the problem of multi-branch
        # models in case of multi-output models without -1 lookback usage (it was like training parallel uncorrelated models for each branch)
        if self.cell_index in self.network_build_info.used_cell_indexes:
            input_name = 'input' if self.cell_index == 0 else f'cell_{self.prev_cell_index}'
            partitions_dict[f'{input_name} -> cell_{self.cell_index}'] = self._compute_partition_size(inputs)

            cell_output = self._build_cell(filters, reduction, inputs)
            self.prev_cell_index = self.cell_index

            if self.multi_output:
                # use a dropout rate which is proportional to the cell index
                drop_rate = round(self.dropout_prob * ((self.cell_index + 1) / self.total_cells), 2)
                self._generate_output(cell_output, dropout_prob=drop_rate)
        # skip cell creation, since it will not be used
        else:
            cell_output = None

        self.cell_index += 1

        # skip and last output, last previous output becomes the skip output for the next cell (from -1 to -2),
        # while -1 is the output of the created cell
        return [inputs[-1], cell_output]

    @abstractmethod
    def _generate_output(self, input_tensor: tf.Tensor, dropout_prob: float = 0.0) -> tf.Tensor:
        '''
        Generate the layers necessary to produce the target output of the network.
        These layers can be stacked after any cell output to get the final output.
        They are always used at the network end, but could also be used for intermediate outputs.

        Args:
            input_tensor: the tensor to process, i.e., any cell output.
            dropout_prob: [0, 1] probability value for applying dropout on the exit.

        Returns:
            a tensor containing an intermediate output or the final output of the network.
        '''
        raise NotImplementedError()

    @abstractmethod
    def build_model(self, cell_spec: 'list[tuple]', add_imagenet_stem: bool = False) -> 'tuple[Model, dict, int]':
        '''
        Build a Keras model from the given cell specification.
        The macro-structure varies between the generators concrete implementations, defining different macro-architectures based on the problem.

        Args:
            cell_spec: the cell specification defining the model motifs.
            add_imagenet_stem: prepend to the network the "ImageNet stem" used in NASNet and PNAS.

        Returns:
            Keras model, partition dictionary, and the final cell index.
        '''
        raise NotImplementedError()

    def _build_cell(self, filters: int, reduction: bool, inputs: 'list[tf.Tensor]'):
        '''
        Generate the cell neural network implementation from the cell encoding.

        Args:
            filters: Number of filters to use
            reduction: if it's a reduction cell or not
            inputs: Tensors that can be used as input (lookback inputs, inputs from previous cells or dataset samples)

        Returns:
            (tf.Tensor): output tensor of the cell
        '''
        # normalize inputs if necessary (different spatial dimension of at least one input, from the one expected by the actual cell)
        if self.lookback_reshape and self.cell_index in self.network_build_info.need_input_norm_indexes:
            inputs = self._normalize_inputs(inputs, filters)

        # else concatenate all the intermediate blocks that compose the cell
        block_outputs = []
        total_inputs = inputs  # initialized with provided previous cell inputs (-1 and -2), but will contain also the block outputs of this cell
        for i, block in enumerate(self.network_build_info.cell_specification):
            self.block_index = i
            block_out = self._build_block(block, filters, reduction, total_inputs)

            # allow fast insertion in order with respect to block creation
            block_outputs.append(block_out)
            # concatenate the two lists to provide the whole inputs available for next blocks of the cell
            total_inputs = block_outputs + inputs

        if self.concat_only_unused:
            block_outputs = [block_outputs[i] for i in self.network_build_info.unused_block_outputs]

        # concatenate and reduce depth to filters value, otherwise cell output would be a (b * filters) tensor depth
        if len(block_outputs) > 1:
            # concatenate all 'Add' layers, outputs of each single block
            if self.drop_path_keep_prob < 1.0:
                cell_ratio = (self.cell_index + 1) / self.total_cells
                total_train_steps = self.training_steps_per_epoch * self.epochs

                sdp = ops.ScheduledDropPath(self.drop_path_keep_prob, cell_ratio, total_train_steps, dims=len(self.input_shape),
                                            name=f'sdp_c{self.cell_index}_concat')(block_outputs)
                concat_layer = layers.Concatenate(axis=-1)(sdp)
            else:
                concat_layer = layers.Concatenate(axis=-1)(block_outputs)
            x = self.op_instantiator.generate_pointwise_conv(filters, strided=False, name=f'concat_pointwise_conv_c{self.cell_index}')
            cell_out = x(concat_layer)
        # avoids concatenation, since it is unnecessary
        else:
            cell_out = block_outputs[0]

        # perform squeeze-excitation (SE) on cell output, if set in the configuration.
        # Squeeze-excitation is performed before residual sum, following SE paper indications.
        if self.se_cell_output:
            cell_out = ops.SqueezeExcitation(self.op_dims, filters, se_ratio=8, use_bias=False, weight_reg=self.l2_weight_reg,
                                             name=f'squeeze_excitation_c{self.cell_index}')(cell_out)

        # if option is set in configuration, make the cell a residual unit, by summing cell output with nearest lookback input
        if self.residual_cells:
            nearest_lookback = max(self.network_build_info.used_lookbacks)
            lb_input = inputs[nearest_lookback]

            # check if a linear projection must be performed (case where input shape and output have different sizes, see Resnet paper)
            lb_shape = lb_input.shape.as_list()
            out_shape = cell_out.shape.as_list()

            # 0 is the batch size, 1 is always related to spatial (for both 1D and 2D)
            different_depth = lb_shape[-1] != out_shape[-1]
            different_spatial = lb_shape[1] != out_shape[1]

            # if shapes are different, apply a linear projection, otherwise use the lookback as it is
            pool_kernel_groups = 'x'.join(['2'] * self.op_dims)
            pool_name = f'{pool_kernel_groups} maxpool'
            # linear projection = max pool + pointwise conv if also different depth (second and third arguments), otherwise just max pool
            if different_spatial:
                lb_input = self.op_instantiator.build_op_layer(pool_name, out_shape[-1], lb_shape[-1],
                                                               f'_residual_proj_c{self.cell_index}', strided=True)(lb_input)
            # linear projection = pointwise conv
            elif different_depth:
                lb_input = self.op_instantiator.generate_pointwise_conv(filters, strided=False,
                                                                        name=f'pointwise_conv_residual_proj_c{self.cell_index}')(lb_input)

            cell_out = layers.Add(name=f'residual_c{self.cell_index}')([cell_out, lb_input])

        return cell_out

    def _normalize_inputs(self, inputs: 'list[tf.Tensor]', filters: int):
        '''
        Normalize tensor dimensions between -2 and -1 inputs if they diverge (both spatial and depth normalization is applied).
        In actual architecture, the normalization should happen only if -2 is a normal cell output and -1 is instead the output
        of a reduction cell.

        Args:
            inputs (list<tf.Tensor>): -2 and -1 input tensors

        Returns:
            (list[tf.Tensor]): updated tensor list
        '''

        # uniform the depth and spatial dimension between the two inputs, using a pointwise convolution
        self._logger.debug("Normalizing inputs' spatial dims (cell %d)", self.cell_index)
        x = self.op_instantiator.generate_pointwise_conv(filters, strided=True, name=f'pointwise_conv_input_c{self.cell_index}')
        # override input with the normalized one
        inputs[-2] = x(inputs[-2])

        return inputs

    def _build_block(self, block_spec: tuple, filters: int, reduction: bool, inputs: 'list[tf.Tensor]'):
        '''
        Generate a block, following PNAS conventions.

        Args:
            block_spec: block specification (in1, op1, in2, op2)
            filters: amount of filters to use in layers
            reduction: if the block belongs to a reduction cell
            inputs: tensors that can be used as inputs for this block

        Returns:
            (tf.tensor): Output of Add keras layer
        '''
        input_L, op_L, input_R, op_R = block_spec
        input_L_shape = inputs[input_L].shape.as_list()
        input_R_shape = inputs[input_R].shape.as_list()
        cur_cell_target_output_shape = self._cur_target_shape()  # has no batch dimension, contrary to inputs

        # in reduction cells, stride only on lookback inputs (first level of cell DAG)
        strided_L = reduction and input_L < 0
        strided_R = reduction and input_R < 0

        # New option to avoid input shape regularization with pointwise conv, when -2 refers to a normal cell and -1 to a reduction cell
        # normal cells can use stride if a "skip" lookback has a bigger dimension than cell target output
        if not self.lookback_reshape:
            strided_L = strided_L or (input_L < -1 and input_L_shape[1] != cur_cell_target_output_shape[0])
            strided_R = strided_R or (input_R < -1 and input_R_shape[1] != cur_cell_target_output_shape[0])

        input_L_depth = input_L_shape[-1]
        input_R_depth = input_R_shape[-1]

        # instantiate a Keras layer for operator, called with the respective input
        name_suffix = f'_c{self.cell_index}b{self.block_index}'
        left_layer = self.op_instantiator.build_op_layer(op_L, filters, input_L_depth, f'{name_suffix}L', strided=strided_L)(inputs[input_L])
        right_layer = self.op_instantiator.build_op_layer(op_R, filters, input_R_depth, f'{name_suffix}R', strided=strided_R)(inputs[input_R])

        if self.drop_path_keep_prob < 1.0:
            cell_ratio = (self.cell_index + 1) / self.total_cells
            total_train_steps = self.training_steps_per_epoch * self.epochs

            sdp = ops.ScheduledDropPath(self.drop_path_keep_prob, cell_ratio, total_train_steps, dims=len(self.input_shape),
                                        name=f'sdp{name_suffix}')([left_layer, right_layer])
            return self.op_instantiator.generate_block_join_operator(name_suffix)(sdp)
        else:
            return self.op_instantiator.generate_block_join_operator(name_suffix)([left_layer, right_layer])

    def define_callbacks(self, tb_logdir: str, score_metric: str) -> 'list[callbacks.Callback]':
        '''
        Define callbacks used in model training.

        Returns:
            Keras callbacks to apply in model.fit() function.
        '''
        # By default shows losses and metrics for both training and validation
        model_callbacks = [callbacks.TensorBoard(log_dir=tb_logdir, profile_batch=0, histogram_freq=0, update_freq='epoch')]

        if self.save_weights:
            # Save best weights, using as metric the last output in case of multi-output models
            target_metric = f'val_Softmax_c{max(self.network_build_info.used_cell_indexes)}_{score_metric}' \
                if self.multi_output and self.network_build_info.blocks > 0 else f'val_{score_metric}'
            model_callbacks.append(callbacks.ModelCheckpoint(filepath=os.path.join(tb_logdir, 'best_weights.h5'),
                                                             save_weights_only=True, save_best_only=True, monitor=target_metric, mode='max'))

        # TODO: if you want to use early stopping, training time should be rescaled for predictor
        # es_callback = callbacks.EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1)

        return model_callbacks

    def define_training_hyperparams_and_metrics(self) -> 'tuple[losses.Loss, Optional[dict[str, float]], optimizers.Optimizer, list[metrics.Metric]]':
        '''
        Get elements to finalize the training procedure of the network and compile the model.
        Mainly, returns a suited loss function and optimizer for the model, plus the metrics to analyze during training.

        Returns:
            loss function, loss weights for each class, optimizer, and a list of metrics to compute during training.
        '''
        if self.multi_output and len(self.output_layers) > 1:
            loss = {}
            loss_weights = {}
            for key in self.output_layers:
                index = int(re.match(r'Softmax_c(\d+)', key).group(1))
                loss.update({key: losses.CategoricalCrossentropy()})
                loss_weights.update({key: 1 / (2 ** (self.total_cells - index))})
        else:
            loss = losses.CategoricalCrossentropy()
            loss_weights = None

        # accuracy is better as string, since it automatically converted to binary or categorical based on loss
        model_metrics = ['accuracy', tfa.metrics.F1Score(num_classes=self.output_classes_count, average='macro')]

        if self.cdr_config['enabled']:
            decay_period = self.training_steps_per_epoch * self.cdr_config['period_in_epochs']
            lr_schedule = optimizers.schedules.CosineDecayRestarts(self.lr, decay_period, self.cdr_config['t_mul'],
                                                                   self.cdr_config['m_mul'], self.cdr_config['alpha'])
            # weight decay for adamW, if used
            wd_schedule = optimizers.schedules.CosineDecayRestarts(self.wr, decay_period, self.cdr_config['t_mul'],
                                                                   self.cdr_config['m_mul'], self.cdr_config['alpha'])
        # if cosine decay restart is not enabled, use cosine decay restart
        else:
            lr_schedule = optimizers.schedules.CosineDecay(self.lr, self.training_steps_per_epoch * self.epochs)
            wd_schedule = optimizers.schedules.CosineDecay(self.wr, self.training_steps_per_epoch * self.epochs)

        optimizer = tfa.optimizers.AdamW(wd_schedule, lr_schedule) if self.use_adamW else optimizers.Adam(learning_rate=lr_schedule)

        return loss, loss_weights, optimizer, model_metrics