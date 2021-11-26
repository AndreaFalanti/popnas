import re

import tensorflow as tf
from tensorflow.keras import layers, regularizers, optimizers, losses, callbacks, Model, Sequential

import log_service
import ops
from utils.func_utils import to_int_tuple, list_flatten, compute_tensor_byte_size


class CellInfo:
    '''
    Helper class that extrapolate and store some relevant info from the cell specification, used for building the actual CNN.
    '''
    def __init__(self, cell_spec: 'list[tuple]') -> None:
        self.specification = cell_spec
        # it's a list of tuples, so already grouped by 4
        self.blocks = len(cell_spec)

        flat_cell_spec = list_flatten(cell_spec)
        # take only BLOCK input indexes (list even indices, discard -1 and -2), eliminating duplicates
        used_block_outputs = set(filter(lambda el: el >= 0, flat_cell_spec[::2]))
        self.used_lookbacks = set(filter(lambda el: el < 0, flat_cell_spec[::2]))
        self.unused_block_outputs = [x for x in range(0, self.blocks) if x not in used_block_outputs]
        self.use_skip = self.used_lookbacks.issuperset({-2})


class ModelGenerator:
    '''
    Class used to build a CNN Keras model, given a cell specification.
    '''

    # TODO: missing max_lookback to adapt inputs based on the actual lookback. For now only 1 or 2 is supported. Also, lookforward is not supported.
    def __init__(self, lr: float, filters: int, weight_norm: float, normal_cells_per_motif: int, motifs: int, drop_path_prob: int,
                 epochs: int, training_steps_per_epoch: int, concat_only_unused: bool = True, data_augmentation_model: Sequential = None):
        self._logger = log_service.get_logger(__name__)
        self.op_regexes = self.__compile_op_regexes()

        self.concat_only_unused = concat_only_unused
        self.lr = lr
        self.filters = filters
        self.motifs = motifs
        self.normal_cells_per_motif = normal_cells_per_motif
        self.total_cells = motifs * (normal_cells_per_motif + 1) - 1

        self.weight_norm = regularizers.l2(weight_norm) if weight_norm is not None else None
        self.drop_path_keep_prob = 1.0 - drop_path_prob

        # necessary for techniques that scale parameters during training, like cosine decay and scheduled drop path
        self.epochs = epochs
        self.training_steps_per_epoch = training_steps_per_epoch

        # if not None, data augmentation will be integrated in the model to be performed directly on the GPU
        self.data_augmentation_model = data_augmentation_model

        # attributes defined below are manipulated and used during model building.
        # defined in class to avoid having lots of parameter passing in each function.

        # used for layers naming
        self.cell_index = 0
        self.block_index = 0

        # for depth adaptation purposes
        self.prev_cell_filters = 0

        # info about the actual cell processed
        # noinspection PyTypeChecker
        self.cell = None  # type: CellInfo

    def __compile_op_regexes(self):
        '''
        Build a dictionary with compiled regexes for each parametrized supported operation.

        Returns:
            (dict): Regex dictionary
        '''
        return {'conv': re.compile(r'(\d+)x(\d+) conv'),
                'dconv': re.compile(r'(\d+)x(\d+) dconv'),
                'tconv': re.compile(r'(\d+)x(\d+) tconv'),
                'stack_conv': re.compile(r'(\d+)x(\d+)-(\d+)x(\d+) conv'),
                'pool': re.compile(r'(\d+)x(\d+) (max|avg)pool')}

    def __compute_partition_size(self, cell_inputs: 'list[tf.Tensor]'):
        input_tensors_size = 0

        for lb in self.cell.used_lookbacks:
            input_tensors_size += compute_tensor_byte_size(cell_inputs[lb])

        return input_tensors_size

    def __adjust_partitions(self, partitions_dict: dict, output_tensor: tf.Tensor):
        # adjust partition sizes, by replacing the last element with only the last cell output, since only that is passed
        # to GAP (no -2, even if used in cell specification).
        if len(partitions_dict['sizes']) > 1:
            partitions_dict['sizes'][-1] = compute_tensor_byte_size(output_tensor)
            # remove skipped cells from feasible partitions, when using only skip inputs
            if partitions_dict['use_skip_only']:
                partitions_dict['sizes'][1:-1] = partitions_dict['sizes'][2:-1:2]
        # initial thrust case, use append since no partitions have been made (only input size is present)
        else:
            partitions_dict['sizes'].append(compute_tensor_byte_size(output_tensor))

    def __build_cell_util(self, filters: int, inputs: list, partitions_dict: dict, reduction: bool = False):
        '''
        Simple helper function for building a cell and quickly return the inputs for next cell.

        Args:
            filters (int): number of filters
            inputs (list<tf.Tensor>): previous cells output, that can be used as inputs in the current cell
            reduction (bool, optional): Build a reduction cell? Defaults to False.

        Returns:
            (list<tf.Tensor>): Usable inputs for next cell
        '''

        stride = (2, 2) if reduction else (1, 1)
        adapt_depth = filters != self.prev_cell_filters

        cell_output = self.__build_cell(filters, stride, inputs, adapt_depth)
        self.cell_index += 1
        self.prev_cell_filters = filters

        # skip and last output, last previous output becomes the skip output for the next cell (from -1 to -2),
        # while -1 is the output of the created cell
        new_inputs = [inputs[-1], cell_output]

        partitions_dict['sizes'].append(self.__compute_partition_size(new_inputs))
        return new_inputs

    def build_model(self, cell_spec: 'list[tuple]'):
        self.cell = CellInfo(cell_spec)

        if len(cell_spec) > 0:
            M = self.motifs
            N = self.normal_cells_per_motif
        # initial thrust case, empty cell
        else:
            M = 0
            N = 0

        # store partition sizes (computed between each two adjacent cells and between last cell and GAP
        partitions_dict = {
            'sizes': [],
            'use_skip_only': self.cell.used_lookbacks.issubset({-2})
        }

        filters = self.filters
        # reset indexes
        self.cell_index = 0
        self.block_index = 0

        # TODO: dimensions are unknown a priori (None), but could be inferred by dataset used
        # TODO: dims are required for inputs normalization, hardcoded for now
        model_input = layers.Input(shape=(32, 32, 3))
        # save initial input size in bytes into partition list
        partitions_dict['sizes'].append(compute_tensor_byte_size(model_input))
        # put prev filters = input depth
        self.prev_cell_filters = 3

        # define inputs usable by blocks
        # last_output will be the input image at start, while skip_output is set to None to trigger
        # a special case in build_cell (avoids input normalization)
        if self.data_augmentation_model is None:
            cell_inputs = [None, model_input]  # [skip, last]
        # data augmentation integrated in the model to perform it in GPU, input is therefore the output of the data augmentation model
        else:
            data_augmentation = self.data_augmentation_model(model_input)
            cell_inputs = [None, data_augmentation]  # [skip, last]

        # add (M - 1) times N normal cells and a reduction cell
        for motif_index in range(M):
            # add N times a normal cell
            for _ in range(N):
                cell_inputs = self.__build_cell_util(filters, cell_inputs, partitions_dict)

            # add 1 time a reduction cell, except for last motif
            if motif_index + 1 < M:
                filters = filters * 2
                cell_inputs = self.__build_cell_util(filters, cell_inputs, partitions_dict, reduction=True)

        # take last cell output and use it in GAP
        last_output = cell_inputs[-1]

        self.__adjust_partitions(partitions_dict, last_output)

        gap = layers.GlobalAveragePooling2D(name='GAP')(last_output)
        # TODO: other datasets have a different number of classes, should be a parameter (10 as constant is bad)
        output = layers.Dense(10, activation='softmax', name='Softmax', kernel_regularizer=self.weight_norm)(gap)  # only logits

        return Model(inputs=model_input, outputs=output), partitions_dict

    def __build_cell(self, filters, stride, inputs, adapt_depth: bool):
        '''
        Generate cell from action list. Following PNAS paper, addition is used to combine block results.

        Args:
            filters (int): Initial filters to use
            stride (tuple<int, int>): (1, 1) for normal cells, (2, 2) for reduction cells
            inputs (list<tf.tensor>): Possible tensors to use as input (based on action_list index value)
            adapt_depth (bool): True if there is a change in filter size, operations that don't alter depth must be addressed accordingly.

        Returns:
            (tf.Tensor): output tensor of the cell
        '''
        # normalize inputs if necessary (also avoid normalization if model doesn't use -2 input)
        # TODO: a refactor could also totally remove -2 from inputs in this case
        if self.cell.use_skip:
            inputs = self.__normalize_inputs(inputs)

        # else concatenate all the intermediate blocks that compose the cell
        block_outputs = []
        total_inputs = inputs  # initialized with provided previous cell inputs (-1 and -2), but will contain also the block outputs of this cell
        for i, block in enumerate(self.cell.specification):
            self.block_index = i
            block_out = self.__build_block(block, filters, stride, total_inputs, adapt_depth)

            # allow fast insertion in order with respect to block creation
            block_outputs.append(block_out)
            # concatenate the two lists to provide the whole inputs available for next blocks of the cell
            total_inputs = block_outputs + inputs

        if self.concat_only_unused:
            block_outputs = [block_outputs[i] for i in self.cell.unused_block_outputs]

        # concatenate and reduce depth to filters value, otherwise cell output would be a (b * filters) tensor depth
        if len(block_outputs) > 1:
            # concatenate all 'Add' layers, outputs of each single block
            if self.drop_path_keep_prob < 1.0:
                cell_ratio = (self.cell_index + 1) / self.total_cells
                total_train_steps = self.training_steps_per_epoch * self.epochs

                sdp = ops.ScheduledDropPath(self.drop_path_keep_prob, cell_ratio, total_train_steps,
                                            name=f'sdp_c{self.cell_index}_concat')(block_outputs)
                concat_layer = layers.Concatenate(axis=-1)(sdp)
            else:
                concat_layer = layers.Concatenate(axis=-1)(block_outputs)
            x = ops.Convolution(filters, (1, 1), (1, 1))
            x._name = f'concat_pointwise_conv_c{self.cell_index}'
            return x(concat_layer)
        # avoids also concatenation, since it is unnecessary
        else:
            return block_outputs[0]

    def __normalize_inputs(self, inputs):
        '''
        Normalize tensor dimensions between -2 and -1 inputs if they diverge (either spatial and depth).
        In actual architecture the normalization should happen only if -2 is a normal cell output and -1 is instead the output
        of a reduction cell (and in second cell because -2 is the starting image input).

        Args:
            inputs (list<tf.Tensor>): -2 and -1 input tensors

        Returns:
            [list<tf.Tensor>]: updated tensor list (input list could be unchanged, but the list will be returned anyway)
        '''

        # Initial cell case, skip input is not defined, simply use the other input without any depth normalization
        if inputs[-2] is None:
            inputs[-2] = inputs[-1]
            return inputs

        skip_depth = inputs[-2].get_shape().as_list()[3]
        last_depth = inputs[-1].get_shape().as_list()[3]

        # by checking either height or width it's possible to check if tensor dim between the last two cells outputs diverges (normal and reduction)
        # values are None if no starting dimension is set, so make sure to have dimensions set in the network input layer
        skip_height = inputs[-2].get_shape().as_list()[1]
        last_height = inputs[-1].get_shape().as_list()[1]

        # address cases where the tensor dims of the two inputs diverge
        # spatial dim divergence, pointwise convolution with (2, 2) stride to reduce dimensions of normal cell input to reduce one
        if skip_height != last_height:
            # also uniform the depth between the two inputs
            self._logger.debug("Normalizing inputs' spatial dims (cell %d)", self.cell_index)
            x = ops.Convolution(last_depth, (1, 1), strides=(2, 2))
            x._name = f'pointwise_conv_input_c{self.cell_index}'
            # override input with the normalized one
            inputs[-2] = x(inputs[-2])
        # only depth divergence, should not happen with actual algorithm (should always be both spatial and depth if dims diverge)
        # TODO: also it is no more required and could be not good for the network
        elif skip_depth != last_depth:
            self._logger.debug("Normalizing inputs' depth (cell %d)", self.cell_index)
            x = ops.Convolution(last_depth, (1, 1), strides=(1, 1))  # no stride
            x._name = f'pointwise_conv_input_c{self.cell_index}'
            inputs[-2] = x(inputs[-2])

        return inputs

    def __build_block(self, block_spec: tuple, filters: int, stride: 'tuple(int, int)', inputs: list, adapt_depth: bool):
        '''
        Generate a block, following PNAS conventions.

        Args:
            block_spec: [description]
            filters: [description]
            stride: [description]
            inputs: [description]
            adapt_depth: [description]

        Returns:
            (tf.tensor): Output of Add keras layer
        '''
        input_L, op_L, input_R, op_R = block_spec

        # in reduction cell, still use stride (1, 1) if not using "original inputs" (-1, -2, no reduction for other blocks' outputs)
        stride_L = stride if input_L < 0 else (1, 1)
        stride_R = stride if input_R < 0 else (1, 1)

        # parse_action returns a custom layer model, that is then called with chosen input
        left_layer = self.__build_layer(filters, op_L, adapt_depth, strides=stride_L, tag='L')(inputs[input_L])
        right_layer = self.__build_layer(filters, op_R, adapt_depth, strides=stride_R, tag='R')(inputs[input_R])

        if self.drop_path_keep_prob < 1.0:
            cell_ratio = (self.cell_index + 1) / self.total_cells
            total_train_steps = self.training_steps_per_epoch * self.epochs
            sdp = ops.ScheduledDropPath(self.drop_path_keep_prob, cell_ratio, total_train_steps,
                                        name=f'sdp_c{self.cell_index}b{self.block_index}')([left_layer, right_layer])
            return layers.Add()(sdp)
        else:
            return layers.Add()([left_layer, right_layer])

    def __build_layer(self, filters, operator, adapt_depth: bool, strides=(1, 1), tag='L'):
        '''
        Generate correct custom layer for provided action. Certain cases are handled incorrectly,
        so that model can still be built, albeit not with original specification

        # Args:
            filters: number of filters
            operator: operator to use
            adapt_depth (bool): adapt depth of operators that don't alter it
            strides: stride to reduce spatial size
            tag (string): either L or R, identifying the block operation

        # Returns:
            (tf.keras.Model): The custom layer corresponding to the action (see ops.py)
        '''

        # check non parametrized operations first since they don't require a regex and are faster
        if operator == 'identity':
            # 'identity' action case, if using (2, 2) stride it's actually handled as a pointwise convolution
            if strides == (2, 2):
                model_name = f'pointwise_conv_c{self.cell_index}b{self.block_index}{tag}'
                x = ops.Convolution(filters, kernel=(1, 1), strides=strides, name=model_name, weight_norm=self.weight_norm)
                return x
            else:
                # else just submits a linear layer if shapes match
                model_name = f'identity_c{self.cell_index}b{self.block_index}{tag}'
                x = ops.Identity(filters, name=model_name)
                return x

        # check for separable conv
        match = self.op_regexes['dconv'].match(operator)  # type: re.Match
        if match:
            model_name = f'{match.group(1)}x{match.group(2)}_dconv_c{self.cell_index}b{self.block_index}{tag}'
            x = ops.SeparableConvolution(filters, kernel=to_int_tuple(match.group(1, 2)), strides=strides,
                                         name=model_name, weight_norm=self.weight_norm)
            return x

        # check for transpose conv
        match = self.op_regexes['tconv'].match(operator)  # type: re.Match
        if match:
            model_name = f'{match.group(1)}x{match.group(2)}_tconv_c{self.cell_index}b{self.block_index}{tag}'
            x = ops.TransposeConvolution(filters, kernel=to_int_tuple(match.group(1, 2)), strides=strides,
                                         name=model_name, weight_norm=self.weight_norm)
            return x

        # check for stacked conv operation
        match = self.op_regexes['stack_conv'].match(operator)  # type: re.Match
        if match:
            f = [filters, filters]
            k = [to_int_tuple(match.group(1, 2)), to_int_tuple(match.group(3, 4))]
            s = [strides, (1, 1)]

            model_name = f'{match.group(1)}x{match.group(2)}-{match.group(3)}x{match.group(4)}_conv_c{self.cell_index}b{self.block_index}{tag}'
            x = ops.StackedConvolution(f, k, s, name=model_name, weight_norm=self.weight_norm)
            return x

        # check for standard conv
        match = self.op_regexes['conv'].match(operator)  # type: re.Match
        if match:
            model_name = f'3x3_conv_c{self.cell_index}b{self.block_index}{tag}'
            x = ops.Convolution(filters, kernel=to_int_tuple(match.group(1, 2)), strides=strides,
                                name=model_name, weight_norm=self.weight_norm)
            return x

        # check for pooling
        match = self.op_regexes['pool'].match(operator)  # type: re.Match
        if match:
            size = to_int_tuple(match.group(1, 2))
            pool_type = match.group(3)

            model_name = f'{match.group(1)}x{match.group(2)}_{pool_type}pool_c{self.cell_index}b{self.block_index}{tag}'
            x = ops.PoolingConv(filters, pool_type, size, strides, name=model_name, weight_norm=self.weight_norm) if adapt_depth \
                else ops.Pooling(pool_type, size, strides, name=model_name)

            return x

        raise ValueError('Operation not covered by POPNAS algorithm')

    def define_callbacks(self, tb_logdir: str):
        '''
        Define callbacks used in model training.

        Returns:
            (tf.keras.Callback[]): Keras callbacks
        '''
        # By default shows losses and metrics for both training and validation
        tb_callback = callbacks.TensorBoard(log_dir=tb_logdir, profile_batch=0, histogram_freq=0, update_freq='epoch')

        # TODO: if you want to use early stopping, training time should be rescaled for predictor
        # es_callback = callbacks.EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1)

        return [tb_callback]

    def define_training_hyperparams_and_metrics(self):
        loss = losses.CategoricalCrossentropy()
        metrics = ['accuracy']

        # TODO: perform more tests on learning rate schedules and optimizers, for now ADAM with cosineDecayRestart seems to do better on 20 epochs
        schedule = optimizers.schedules.CosineDecayRestarts(self.lr, self.training_steps_per_epoch * 3)
        # schedule_2 = optimizers.schedules.CosineDecay(self.lr, self.training_steps_per_epoch * self.epochs)
        # sgdr_optimizer = optimizers.SGD(learning_rate=schedule_2, momentum=0.9)
        adam_optimizer = optimizers.Adam(learning_rate=schedule)

        return loss, adam_optimizer, metrics
