import os.path
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Type, NamedTuple, Union

import tensorflow as tf
from tensorflow.keras import layers, regularizers, optimizers, activations, losses, metrics, callbacks, Model, Sequential

import log_service
import models.operators.layers as ops
import utils.tensor_utils as tu
from models.graphs.network_graph import NetworkGraph
from models.operators.op_instantiator import OpInstantiator
from models.optimizers import instantiate_optimizer_and_schedulers
from models.results.base import BaseTrainingResults
from search_space import CellSpecification
from utils.config_dataclasses import TrainingHyperparametersConfig, ArchitectureHyperparametersConfig
from utils.func_utils import elementwise_mult
from utils.nn_utils import support_weight_decay


class WrappedTensor(NamedTuple):
    '''
    Wrapper for a TF tensor, which is associated to a shape represented with a list containing a number for each dimension (no batch size).

    The shape could be the actual size, but usually it contains multipliers of the input size per dimension
    (e.g. [0.5, 0.5, 4] means that the shape has half the spatial dimension of the input image, but 4 times the number of channels).
    In this way, fully convolutional NN can set the input spatial dimensions to None, but still keep track of when the tensor shapes are altered.
    '''
    tensor: tf.Tensor
    shape: 'list[float, ...]'


@dataclass
class NetworkBuildInfo:
    '''
    Helper class storing relevant info extracted from the cell specification, used for building the actual neural network model.
    '''
    cell_spec: CellSpecification
    blocks: int
    used_cell_indexes: 'list[int]'
    reduction_cell_indexes: 'list[int]'
    need_input_norm_indexes: 'list[int]'


class BaseModelGenerator(ABC):
    '''
    Abstract class used as a baseline for model generators concrete implementations.
    This class contains all the shared logic between the models, like how the blocks and cells are built,
    while the macro-implementation, loss function, optimizer, and metrics can be defined ad-hoc for the type of tasks addressed.

    For fully convolution neural networks, the input shape can have the spatial dimensions set to None, but the number of channels must be provided.
    '''

    # TODO: missing max_lookback to adapt inputs based on the actual lookback. For now only 1 or 2 is supported.
    def __init__(self, train_hp: TrainingHyperparametersConfig, arc_hp: ArchitectureHyperparametersConfig,
                 training_steps_per_epoch: int, output_classes_count: int, input_shape: 'tuple[int, ...]',
                 data_augmentation_model: Optional[Sequential] = None, preprocessing_model: Optional[Sequential] = None, save_weights: bool = False):
        self._logger = log_service.get_logger(__name__)

        self.concat_only_unused = arc_hp.concat_only_unused_blocks
        self.lookback_reshape = arc_hp.lookback_reshape
        self.filters = arc_hp.filters
        self.motifs = arc_hp.motifs
        self.normal_cells_per_motif = arc_hp.normal_cells_per_motif
        self.multi_output = arc_hp.multi_output
        self.residual_cells = arc_hp.residual_cells
        self.se_cell_output = arc_hp.se_cell_output
        self.output_classes_count = output_classes_count
        self.input_shape = input_shape
        # basically, it indicates the application domain of the operators, as int (2D for images, 1D for time series)
        self.op_dims = len(input_shape) - 1

        self.lr = train_hp.learning_rate
        self.wd = train_hp.weight_decay
        self.l2_weight_reg = regularizers.l2(self.wd) if (self.wd is not None and not support_weight_decay(train_hp.optimizer.type)) else None
        self.drop_path_keep_prob = 1.0 - train_hp.drop_path
        self.dropout_prob = train_hp.softmax_dropout  # dropout probability on final softmax
        self.optimizer_config = train_hp.optimizer
        self.label_smoothing = train_hp.label_smoothing
        self.filters_ratio = self.filters / self.input_shape[-1]  # the filters expansion ratio applied between input and first layer

        self.save_weights = save_weights

        # necessary for techniques that scale parameters during training, like cosine decay and scheduled drop path
        self.epochs = train_hp.epochs
        self.training_steps_per_epoch = training_steps_per_epoch

        # if not None, data augmentation will be integrated in the model to be performed directly on the GPU
        self.data_augmentation_model = data_augmentation_model
        # if not None, some preprocessing is performed directly in the model (parametric layers like normalization, so that work also for test split)
        self.preprocessing_model = preprocessing_model

        # compute output shapes of each cell, to enable easier tensor shape management
        self.cell_output_shapes = self._compute_cell_output_shapes()

        # op instantiator takes care of handling the instantiation of Keras layers for building the final architecture
        self.activation_f = activations.get(arc_hp.activation_function)
        self.op_instantiator = OpInstantiator(len(input_shape), arc_hp.block_join_operator,
                                              weight_reg=self.l2_weight_reg, activation_f=self.activation_f)

        # attributes defined below this comment are manipulated and used during model building.
        # defined in class to avoid having lots of parameters passing in each function.

        # used for layer naming and partition dictionary
        self.cell_index = 0
        self.block_index = 0

        # info about the actual cell processed and current model outputs
        # noinspection PyTypeChecker
        self.network_build_info = None  # type: NetworkBuildInfo
        self.output_layers = {}

    def get_maximum_cells(self):
        ''' Returns the maximum number of cells that can be stacked in the macro-architecture.'''
        return self.motifs * (self.normal_cells_per_motif + 1) - 1

    def _get_cell_ratio(self, cell_index: int):
        ''' Returns a value in [0, 1], proportional to the cell position in the macro-architecture. '''
        # +1 since cells are 0-indexed
        return (cell_index + 1) / self.get_maximum_cells()

    def get_real_cell_depth(self, cell_spec: CellSpecification) -> int:
        '''
        Compute the real number of cells stacked in a CNN, based on the cell specification and the actual cell stack target.
        Usually, the number of cells stacked in a CNN is the target imposed, but if the inputs use only lookback input values < -1,
        then some of them are actually skipped, leading to a CNN with fewer cells than the imposed number.

        Args:
            cell_spec: the cell specification

        Returns:
            the number of cells stacked in the architecture
        '''
        lookback_inputs = [inp for inp in cell_spec.inputs if inp is not None and inp < 0]
        nearest_lookback = max(lookback_inputs)
        cell_indexes = list(range(self.get_maximum_cells()))

        return len([c_index for c_index in cell_indexes[::nearest_lookback]])

    def alter_macro_structure(self, m: int, n: int, f: int):
        self.motifs = m
        self.normal_cells_per_motif = n
        self.filters = f
        self.filters_ratio = f / self.input_shape[-1]

        # recompute properties associated to the macro-structure parameters
        self.cell_output_shapes = self._compute_cell_output_shapes()

    @abstractmethod
    def _generate_network_info(self, cell_spec: CellSpecification, use_stem: bool) -> NetworkBuildInfo:
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

    def _current_target_shape(self):
        return self.cell_output_shapes[self.cell_index]

    def compute_network_partitions(self, cell_spec: CellSpecification, tensor_dtype: tf.dtypes.DType = tf.float32):
        # empty cell has no partitions
        if cell_spec.is_empty_cell():
            return {}

        # avoid the computation of partitions when the input shape is not fixed (fully convolutional NN can work with arbitrary sizes)
        if any(el is None for el in self.input_shape):
            self._logger.debug('Partitions not computed since the input shape is not fixed')
            return {}

        network_info = self._generate_network_info(cell_spec, use_stem=False)
        abs_furthest_lookback = abs(min(network_info.cell_spec.used_lookbacks))
        output_shapes = ([list(self.input_shape)] * abs_furthest_lookback) + self.cell_output_shapes

        dtype_sizes = {
            tf.float16: 2,
            tf.float32: 4,
            tf.float64: 8,
            tf.int32: 4,
            tf.int64: 8
        }
        dtype_size = dtype_sizes[tensor_dtype]

        partitions_dict = {}
        previous_name = 'input'
        for cell_index in sorted(network_info.used_cell_indexes):
            current_name = f'cell_{cell_index}'
            partitions_dict[f'{previous_name} -> {current_name}'] = \
                sum(tu.compute_bytes_from_tensor_shape(output_shapes[lb + abs_furthest_lookback + cell_index], dtype_size)
                    for lb in network_info.cell_spec.used_lookbacks)
            previous_name = current_name

        return partitions_dict

    @abstractmethod
    def get_output_layers_name(self) -> str:
        ''' Return the name used in the output layers of this model generator. '''
        raise NotImplementedError()

    @abstractmethod
    def _generate_output(self, hidden_tensor: WrappedTensor, dropout_prob: float = 0.0) -> tf.Tensor:
        '''
        Generate the layers necessary to produce the target output of the network from a hidden layer.
        These layers can be stacked after any cell output to get the final output.
        They are always used at the network end, but could also be used for intermediate outputs.

        NOTE: always separate logits from Softmax (or other activation layer), casting the final layer to dtype=float32.
        This makes possible to correctly apply mixed precision, when set in the POPNAS config.

        Args:
            hidden_tensor: the tensor to process, i.e., any cell output.
            dropout_prob: [0, 1] probability value for applying dropout on the exit.

        Returns:
            a tensor containing an intermediate output or the final output of the network.
        '''
        raise NotImplementedError()

    def _get_output_names(self) -> 'list[str]':
        '''
        Return the name of the output layers in multi-output models.
        If there is a single output, return an empty string (easier to process Keras history in this way).
        '''
        return list(self.output_layers.keys()) if len(self.output_layers.keys()) > 1 else ['']

    def _get_macro_params(self, cell_spec: CellSpecification) -> 'tuple[int, int]':
        ''' Returns M and N, setting them to 0 if the cell specification is the empty cell. '''
        return (0, 0) if cell_spec.is_empty_cell() else (self.motifs, self.normal_cells_per_motif)

    def _reset_metadata_indexes(self):
        ''' Reset metadata indexes associated to cells and blocks. '''
        self.cell_index = 0
        self.block_index = 0

    def _apply_preprocessing_and_augmentation(self, model_input: tf.Tensor) -> tf.Tensor:
        '''
        Apply the preprocessing and data augmentation Keras models, stacking them to the model input.
        The preprocessing is usually done in the dataset pipeline, but procedures like normalization are better integrated in the model.
        The augmentation is usually done on CPU, but could be integrated in the model (GPU) if the related configuration flag is set to True.

        Args:
            model_input: the initial model input

        Returns:
            the input tensor itself, or its modification due to the preprocessing and augmentation models, when defined.
        '''
        initial_lookback_input = model_input
        # some data preprocessing is integrated in the model. Update lookback inputs to be the output of the preprocessing model.
        if self.preprocessing_model is not None:
            initial_lookback_input = self.preprocessing_model(initial_lookback_input)
        # data augmentation integrated in the model to perform it in GPU. Update lookback inputs to be the output of the data augmentation model.
        if self.data_augmentation_model is not None:
            initial_lookback_input = self.data_augmentation_model(initial_lookback_input)

        return initial_lookback_input

    # TODO: adapt it for 1D (time-series)
    def _prepend_imagenet_stem(self, cell_inputs: 'list[WrappedTensor]', filters: int) -> 'tuple[list[WrappedTensor], int]':
        '''
        Add a stride-2 3x3 convolution layer followed by 2 reduction cells at the start of the network.
        The stem originates from PNAS and NASNet works, which used this stack of layers to quickly reduce the input dimensionality when
        using larger images than the ones for which the network was designed.

        Args:
            cell_inputs:
            filters:

        Returns:
            the new tensors usable as lookbacks, and the new number of filters.
        '''
        model_input = cell_inputs[-1]
        stem_conv = ops.Convolution(filters, kernel=(3, 3), strides=(2, 2), name='stem_3x3_conv', weight_reg=self.l2_weight_reg)(model_input.tensor)
        new_shape_ratio = elementwise_mult(model_input.shape, [0.5, 0.5, filters / self.input_shape[-1]])
        cell_inputs = [model_input, WrappedTensor(stem_conv, new_shape_ratio)]
        filters = filters * 2
        cell_inputs = self._build_cell_util(filters, cell_inputs, reduction=True)
        filters = filters * 2
        cell_inputs = self._build_cell_util(filters, cell_inputs, reduction=True)

        return cell_inputs, filters

    @abstractmethod
    def build_model(self, cell_spec: CellSpecification, add_imagenet_stem: bool = False) -> 'tuple[Model, list[str]]':
        '''
        Build a Keras model from the given cell specification.
        The macro-structure varies between the generators concrete implementations, defining different macro-architectures based on the problem.

        Args:
            cell_spec: the cell specification defining the model motifs
            add_imagenet_stem: prepend to the network the "ImageNet stem" used in NASNet and PNAS

        Returns:
            Keras model, and the name of the output layers.
        '''
        raise NotImplementedError()

    @abstractmethod
    def build_model_graph(self, cell_spec: CellSpecification, add_imagenet_stem: bool = False) -> NetworkGraph:
        '''
        Generate the graph representation of the architecture which would be built from the given cell specification.
        The macro-structure varies between the generators concrete implementations, defining different macro-architectures based on the problem.

        The graph can be generated multiple orders of magnitude faster than an actual implementation in TF or Keras, making it possible to analyze
        and compute some metrics in a fast way (i.e., the number of parameters).

        Args:
            cell_spec: the cell specification defining the model motifs
            add_imagenet_stem: prepend to the network the "ImageNet stem" used in NASNet and PNAS

        Returns:
            the NetworkGraph object representing the entire network.
        '''
        raise NotImplementedError()

    def _finalize_model(self, model_input: tf.Tensor, last_cell_output: WrappedTensor) -> Model:
        '''
        Finalize the generation of the Keras functional model.
        It transparently handles both single output and multi output models.

        Args:
            model_input: the initial model input
            last_cell_output: the final output tensor

        Returns:
            the instance of the Keras model built.
        '''
        # outputs are built in _build_cell_util if using multi-output, so just finalize the Model instance in this case.
        # second condition is used to force output building in case of the empty cell, since no cells are present even in multi_output mode.
        if self.multi_output and len(self.output_layers) > 0:
            return Model(inputs=model_input, outputs=self.output_layers.values())
        else:
            output = self._generate_output(last_cell_output, self.dropout_prob)
            return Model(inputs=model_input, outputs=output)

    def _build_cell_util(self, filters: int, inputs: 'list[WrappedTensor]', reduction: bool = False) -> 'list[WrappedTensor]':
        '''
        Simple helper function for building a cell and quickly return the lookback inputs for the next cell.

        Args:
            filters: number of filters
            inputs: previous cells output, that can be used as inputs in the current cell (lookback inputs)
            reduction: set it to true in reduction cells

        Returns:
            lookback inputs for the next cell
        '''
        # check that this cell is actually connected to the final output layer, otherwise skips its generation.
        # in this way, multi-output models using only -2 lookback won't have separate uncorrelated branches with separate outputs,
        # which is not the desired behavior.
        if self.cell_index in self.network_build_info.used_cell_indexes:
            cell_output = self._build_cell(filters, reduction, inputs)

            if self.multi_output:
                # use a dropout rate which is proportional to the cell index
                drop_rate = round(self.dropout_prob * self._get_cell_ratio(self.cell_index), 2)
                self._generate_output(cell_output, dropout_prob=drop_rate)
        else:
            cell_output = None

        self.cell_index += 1

        # skip and last output, last previous output becomes the skip output for the next cell (from -1 to -2),
        # while -1 is the output of the created cell
        return [inputs[-1], cell_output]

    def _build_cell(self, filters: int, reduction: bool, inputs: 'list[WrappedTensor]') -> WrappedTensor:
        '''
        Generate the cell neural network implementation from the cell encoding.

        Args:
            filters: Number of filters to use
            reduction: if it is a reduction cell or not
            inputs: Tensors that can be used as input (lookback inputs, inputs from previous cells or dataset samples)

        Returns:
            output tensor of the cell
        '''

        out_shape = self._current_target_shape()

        # upsample when necessary (the spatial ratio between target and input is >= 2),
        # since there is no way for the 99% of operators to upsample in a native way
        inputs = [self._upsample_tensor_if_necessary(lb, out_shape, filters, name=f'transpose_conv_lb{len(inputs) - i}_upsample_c{self.cell_index}')
                  for i, lb in enumerate(inputs)]

        # normalize inputs if necessary (different spatial dimension of at least one input, from the one expected by the actual cell)
        if self.lookback_reshape and self.cell_index in self.network_build_info.need_input_norm_indexes:
            inputs = self._normalize_inputs(inputs, filters)

        block_outputs = []
        # initialized with provided lookback inputs (-1 and -2), but will contain also the block outputs of this cell
        total_inputs = inputs
        for i, block in enumerate(self.network_build_info.cell_spec):
            self.block_index = i
            block_out = self._build_block(block, filters, reduction, total_inputs)
            block_outputs.append(block_out)

            # concatenate the two lists to provide the whole inputs available for the next blocks of the cell
            total_inputs = block_outputs + inputs

        if self.concat_only_unused:
            block_outputs = [block_outputs[i] for i in self.network_build_info.cell_spec.unused_blocks]

        # concatenate and reduce depth to the number of filters, otherwise cell output would be a (b * filters) tensor depth
        cell_out = self._concatenate_blocks_into_cell_output(block_outputs, filters)

        # perform squeeze-excitation (SE) on cell output, if set in the configuration.
        # Squeeze-excitation is performed before the residual sum, following SE paper indications.
        if self.se_cell_output:
            cell_out = ops.SqueezeExcitation(self.op_dims, filters, se_ratio=8, use_bias=False, weight_reg=self.l2_weight_reg,
                                             activation_f=self.activation_f, name=f'squeeze_excitation_c{self.cell_index}')(cell_out)

        # if the relative option is set in configuration, make the cell a residual unit, by summing cell output with nearest lookback input
        if self.residual_cells:
            cell_out = self._make_cell_output_residual(cell_out, filters, inputs, out_shape)

        return WrappedTensor(cell_out, out_shape)

    def _make_cell_output_residual(self, cell_out: tf.Tensor, filters: int, lookback_inputs: 'list[WrappedTensor]',
                                   out_shape: 'list[int, ...]') -> tf.Tensor:
        '''
        Sum the nearest used lookback input to the cell output, making a residual connection.
        If the shaped differ, a "linear projection" is applied to the input to reshape it.
        '''
        nearest_lookback = max(self.network_build_info.cell_spec.used_lookbacks)
        lb_tensor, lb_shape = lookback_inputs[nearest_lookback]

        # check if a linear projection must be performed (case where input shape and output have different sizes, see Resnet paper)
        different_depth = not tu.have_tensors_same_depth(lb_shape, out_shape)
        different_spatial = not tu.have_tensors_same_spatial(lb_shape, out_shape)

        # if shapes are different, apply a linear projection, otherwise use the lookback as it is
        pool_kernel_groups = 'x'.join(['2'] * self.op_dims)
        pool_name = f'{pool_kernel_groups} maxpool'
        # linear projection = max pool + pointwise conv if also different depth (second and third arguments), otherwise just max pool
        if different_spatial:
            lb_tensor = self.op_instantiator.build_op_layer(pool_name, filters, f'_residual_proj_c{self.cell_index}',
                                                            adapt_spatial=True, adapt_depth=different_depth)(lb_tensor)
        # linear projection = pointwise conv
        elif different_depth:
            lb_tensor = self.op_instantiator.generate_pointwise_conv(filters, strided=False,
                                                                     name=f'pointwise_conv_residual_proj_c{self.cell_index}')(lb_tensor)

        return layers.Add(name=f'residual_c{self.cell_index}')([cell_out, lb_tensor])

    def _concatenate_blocks_into_cell_output(self, block_outputs: 'list[WrappedTensor]', filters: int) -> tf.Tensor:
        '''
        Concatenate all given block outputs, and reduce tensor depth to filters value.
        If just a single block is provided, then the concatenation is not performed.
        '''
        block_tensors = [b.tensor for b in block_outputs]

        # concatenate multiple block outputs together
        if len(block_tensors) > 1:
            if self.drop_path_keep_prob < 1.0:
                cell_ratio = self._get_cell_ratio(self.cell_index)
                total_train_steps = self.training_steps_per_epoch * self.epochs

                sdp = ops.ScheduledDropPath(self.drop_path_keep_prob, cell_ratio, total_train_steps, dims=len(self.input_shape),
                                            name=f'sdp_c{self.cell_index}_concat')(block_tensors)
                concat_layer = layers.Concatenate(axis=-1, name=f'concat_c{self.cell_index}')(sdp)
            else:
                concat_layer = layers.Concatenate(axis=-1, name=f'concat_c{self.cell_index}')(block_tensors)
            x = self.op_instantiator.generate_pointwise_conv(filters, strided=False, name=f'concat_pointwise_conv_c{self.cell_index}')
            return x(concat_layer)
        # avoids concatenation of a single block, since it is unnecessary
        else:
            return block_tensors[0]

    def _normalize_inputs(self, inputs: 'list[WrappedTensor]', filters: int) -> 'list[WrappedTensor]':
        '''
        Normalize tensor dimensions between -2 and -1 inputs if they diverge (both spatial and depth normalization is applied).
        In actual architecture, the normalization should happen only if -2 is a normal cell output and -1 is instead the output
        of a reduction cell.

        Args:
            inputs: lookback input tensors

        Returns:
            updated tensor list
        '''

        # uniform the depth and spatial dimension between the two inputs, using a pointwise convolution
        self._logger.debug("Normalizing inputs' spatial dims (cell %d)", self.cell_index)
        pointwise = self.op_instantiator.generate_pointwise_conv(filters, strided=True, name=f'pointwise_conv_input_c{self.cell_index}')
        # override input with the normalized one
        inputs[-2] = WrappedTensor(pointwise(inputs[-2].tensor), tu.alter_tensor_shape(inputs[-2].shape, 0.5, 2))

        return inputs

    def _upsample_tensor_if_necessary(self, t: Optional[WrappedTensor], target_shape: 'list[float]', filters: int,
                                      name: str) -> Optional[WrappedTensor]:
        '''
        Upsample a tensor if its spatial dimensions are lower that the target shape, otherwise it returns the original tensor without modifications.
        The upsample is performed through transpose convolution.

        Args:
            t: the tensor to consider (note: None tensors are also fine)
            target_shape: the target shape for the cell output
            filters: the number of filters used in the cell
            name: name of the eventually created upsample layer

        Returns:
            the upsampled tensor, or simply the original one if the upsample is not required.
        '''
        # in case the tensor is None, just return it immediately
        if t is None:
            return t

        # get the spatial ratio as int, if >= 2 it would mean that an upsample is needed...
        spatial_ratio = round(tu.get_tensors_spatial_ratio(target_shape, t.shape))
        if spatial_ratio > 1:
            self._logger.debug('Generating upsample layer %s (target shape: %s, input_shape: %s)', name, target_shape, t.shape)
            tensor_val = self.op_instantiator.generate_transpose_conv(filters, spatial_ratio, name=name)(t.tensor)
            return WrappedTensor(tensor_val, target_shape)
        # ...otherwise return the tensor as it is, without modifications
        else:
            return t

    def _build_branch(self, input_t: WrappedTensor, op: str, target_output_shape: 'list[int, ...]', filters: int,
                      layer_name_suffix: str) -> tf.Tensor:
        input_tensor, input_shape = input_t

        adapt_spatial = not tu.have_tensors_same_spatial(input_shape, target_output_shape)
        adapt_depth = not tu.have_tensors_same_depth(input_shape, target_output_shape)

        # instantiate a Keras layer for the operator, called with the respective input
        return self.op_instantiator.build_op_layer(op, filters, layer_name_suffix, adapt_spatial, adapt_depth)(input_tensor)

    def _build_block(self, block_spec: tuple, filters: int, reduction: bool, inputs: 'list[WrappedTensor]') -> WrappedTensor:
        '''
        Generate a block, following PNAS conventions.

        Args:
            block_spec: block specification (in1, op1, in2, op2)
            filters: number of filters to use in layers
            reduction: if the block belongs to a reduction cell
            inputs: tensors that can be used as inputs for this block

        Returns:
            Output of Add keras layer
        '''
        input_L, op_L, input_R, op_R = block_spec
        target_output_shape = self._current_target_shape()
        # common name suffix to recognize all layers belonging to this specific block
        layer_name_suffix = f'_c{self.cell_index}b{self.block_index}'

        left_layer = self._build_branch(inputs[input_L], op_L, target_output_shape, filters, f'{layer_name_suffix}L')
        right_layer = self._build_branch(inputs[input_R], op_R, target_output_shape, filters, f'{layer_name_suffix}R')

        # apply the scheduled drop path if enabled
        if self.drop_path_keep_prob < 1.0:
            cell_ratio = self._get_cell_ratio(self.cell_index)
            total_train_steps = self.training_steps_per_epoch * self.epochs

            sdp = ops.ScheduledDropPath(self.drop_path_keep_prob, cell_ratio, total_train_steps, dims=len(self.input_shape),
                                        name=f'sdp{layer_name_suffix}')([left_layer, right_layer])
            out = self.op_instantiator.generate_block_join_operator(layer_name_suffix)(sdp)
        else:
            out = self.op_instantiator.generate_block_join_operator(layer_name_suffix)([left_layer, right_layer])

        return WrappedTensor(out, target_output_shape)

    def define_callbacks(self, model_logdir: str, score_metric: str) -> 'list[callbacks.Callback]':
        '''
        Define callbacks used in model training.

        Returns:
            Keras callbacks to apply in model.fit() function.
        '''
        # By default, shows losses and metrics for both training and validation
        model_callbacks = [callbacks.TensorBoard(log_dir=os.path.join(model_logdir, 'tensorboard'),
                                                 profile_batch=0, histogram_freq=0, update_freq='epoch', write_graph=False)]

        if self.save_weights:
            # Save best weights, using as metric the last output in case of multi-output models
            output_layer_name = self.get_output_layers_name()
            last_used_cell = max(self.network_build_info.used_cell_indexes)
            target_metric = f'val_{output_layer_name}_c{last_used_cell}_{score_metric}' \
                if self.multi_output and self.network_build_info.blocks > 0 else f'val_{score_metric}'
            model_callbacks.append(callbacks.ModelCheckpoint(filepath=os.path.join(model_logdir, 'best_weights.ckpt'),
                                                             save_weights_only=True, save_best_only=True, monitor=target_metric, mode='max'))

        # TODO: if you want to use early stopping, training time should be rescaled for predictor
        # es_callback = callbacks.EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1)

        return model_callbacks

    @abstractmethod
    def _get_loss_function(self) -> losses.Loss:
        raise NotImplementedError()

    @abstractmethod
    def _get_metrics(self) -> 'list[metrics.Metric]':
        raise NotImplementedError()

    def define_training_hyperparams_and_metrics(self) -> 'tuple[Union[losses.Loss, dict[str, losses.Loss]], Optional[dict[str, float]],' \
                                                         ' optimizers.Optimizer, list[metrics.Metric]]':
        '''
        Get elements to finalize the training procedure of the network and compile the model.
        Mainly, returns a suited loss function and optimizer for the model, plus the metrics to analyze during training.

        Returns:
            loss function for each output, loss weights for each output, optimizer, and a list of metrics to compute during training.
        '''
        if self.multi_output and len(self.output_layers) > 1:
            output_name = self.get_output_layers_name()
            loss = {}
            loss_weights = {}

            for key in self.output_layers:
                index = int(re.match(rf'{output_name}_c(\d+)', key).group(1))
                loss.update({key: self._get_loss_function()})
                loss_weights.update({key: 1 / (2 ** (self.get_maximum_cells() - index))})
        else:
            loss = self._get_loss_function()
            loss_weights = None

        # accuracy is better as a string, since it is automatically converted to binary or categorical based on loss
        model_metrics = self._get_metrics()

        optimizer = instantiate_optimizer_and_schedulers(self.optimizer_config, self.lr, self.wd, self.training_steps_per_epoch, self.epochs)

        return loss, loss_weights, optimizer, model_metrics

    @staticmethod
    @abstractmethod
    def get_results_processor_class() -> Type[BaseTrainingResults]:
        raise NotImplementedError()
