import re

from tensorflow.keras import layers, regularizers, optimizers, losses, callbacks, Model

import log_service
import ops
from utils.func_utils import to_int_tuple, list_flatten


class ModelGenerator:

    def __init__(self, cell_spec, filters=24, concat_only_unused=True, weight_norm=None):
        '''
        Utility class to build a CNN with structure provided in the action list.

        # Args:
            actions: list of [input; action] pairs that define the cell.
            filters (int): initial number of filters.
            concat_only_unused (bool): concats only unused states at the end of each cell if true, otherwise concats all blocks output.
        '''

        self._logger = log_service.get_logger(__name__)
        self.op_regexes = self.__compile_op_regexes()

        # it's a list of tuples, so already grouped by 4
        self.B = len(cell_spec)
        # TODO: flat the list to use old logic, with a refactor it could be avoided
        cell_spec = list_flatten(cell_spec)

        self.concat_only_unused = concat_only_unused
        # take only BLOCK input indexes (list even indices, discard -1 and -2), eliminating duplicates
        used_inputs = set(filter(lambda el: el >= 0, cell_spec[::2]))
        self.unused_inputs = [x for x in range(0, self.B) if x not in used_inputs]

        self.use_skip = -2 in cell_spec[::2]

        if len(cell_spec) > 0:
            self.action_list = [x for x in zip(*[iter(cell_spec)] * 2)]  # generate a list of tuples (pairs)
            self.M = 3
            self.N = 2
        else:
            self.action_list = []
            self.M = 1
            self.N = 0

        self.filters = filters
        # for depth adaptation purposes
        self.prev_cell_filters = 0

        self.weight_norm = regularizers.l2(weight_norm) if weight_norm is not None else None

    def __compile_op_regexes(self):
        '''
        Build a dictionary with compiled regexes for each parametrized supported operation.

        Returns:
            (dict): Regex dictionary
        '''
        return {'conv': re.compile(r'(\d+)x(\d+) conv'),
                'dconv': re.compile(r'(\d+)x(\d+) dconv'),
                'stack_conv': re.compile(r'(\d+)x(\d+)-(\d+)x(\d+) conv'),
                'pool': re.compile(r'(\d+)x(\d+) (max|avg)pool')}

    def __build_cell_util(self, filters, inputs, reduction=False):
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

        cell_output = self.__build_cell(self.B, self.action_list, filters, stride, inputs, adapt_depth)
        self.cell_index += 1
        self.prev_cell_filters = filters

        # skip and last output, last previous output becomes the skip output for the next cell (from -1 to -2),
        # while -1 is the output of the created cell
        return [inputs[-1], cell_output]

    def build_model(self):
        # used for layers naming, defined in class to avoid to pass them across multiple functions
        self.cell_index = 0
        self.block_index = 0

        filters = self.filters

        # TODO: dimensions are unknown a priori (None), but could be inferred by dataset used
        # TODO: dims are required for inputs normalization, hardcoded for now
        model_input = layers.Input(shape=(32, 32, 3))
        # put prev filters = input depth
        self.prev_cell_filters = 3

        # define inputs usable by blocks
        # last_output will be the input image at start, while skip_output is set to None to trigger
        # a special case in build_cell (avoids input normalization)
        cell_inputs = [None, model_input]  # [skip, last]

        # add (M - 1) times N normal cells and a reduction cell
        for _ in range(self.M - 1):
            # add N times a normal cell
            for _ in range(self.N):
                cell_inputs = self.__build_cell_util(filters, inputs=cell_inputs)

            # add 1 time a reduction cell
            filters = filters * 2
            cell_inputs = self.__build_cell_util(filters, inputs=cell_inputs, reduction=True)

        # add N time a normal cell
        for _ in range(self.N):
            cell_inputs = self.__build_cell_util(filters, inputs=cell_inputs)

        # take last cell output and use it in GAP
        last_output = cell_inputs[-1]
        gap = layers.GlobalAveragePooling2D(name='GAP')(last_output)
        # TODO: other datasets have a different number of classes, should be a parameter (10 as constant is bad)
        output = layers.Dense(10, activation='softmax', name='Softmax', kernel_regularizer=self.weight_norm)(gap)  # only logits

        return Model(inputs=model_input, outputs=output)

    def __build_cell(self, B, action_list, filters, stride, inputs, adapt_depth: bool):
        '''
        Generate cell from action list. Following PNAS paper, addition is used to combine block results.

        Args:
            B (int): Number of blocks in the cell
            action_list (list<tuple<int, string>): List of tuples of 2 elements -> (input, action_name).
                Input can be either -1 (last cell output) or -2 (skip connection).
            filters (int): Initial filters to use
            stride (tuple<int, int>): (1, 1) for normal cells, (2, 2) for reduction cells
            inputs (list<tf.tensor>): Possible tensors to use as input (based on action_list index value)
            adapt_depth (bool): True if there is a change in filter size, operations that don't alter depth must be addressed accordingly.

        Returns:
            (tf.keras.layers.Add | tf.keras.layers.Concatenate): [description]
        '''
        # normalize inputs if necessary (also avoid normalization if model doesn't use -2 input)
        # TODO: a refactor could also totally remove -2 from inputs in this case
        if self.use_skip:
            inputs = self.__normalize_inputs(inputs)

        # if cell size is 1 block only
        if B == 1:
            return self.__build_block(action_list[0], action_list[1], filters, stride, inputs, adapt_depth)

        # else concatenate all the intermediate blocks that compose the cell
        block_outputs = []
        total_inputs = inputs  # initialized with provided previous cell inputs (-1 and -2), but will contain also the block outputs of this cell
        for i in range(B):
            self.block_index = i
            block_out = self.__build_block(action_list[i * 2], action_list[i * 2 + 1], filters, stride, total_inputs, adapt_depth)

            # allow fast insertion in order with respect to block creation
            block_outputs.append(block_out)
            # concatenate the two lists to provide the whole inputs available for next blocks of the cell
            total_inputs = block_outputs + inputs

        if self.concat_only_unused:
            block_outputs = [block_outputs[i] for i in self.unused_inputs]

        # reduce depth to filters value, otherwise concatenation would lead to (b * filters) tensor depth
        if len(block_outputs) > 1:
            # concatenate all 'Add' layers, outputs of each single block
            concat_layer = layers.Concatenate(axis=-1)(block_outputs)
            x = ops.Convolution(filters, (1, 1), (1, 1))
            x._name = f'concat_pointwise_conv_c{self.cell_index}'
            return x(concat_layer)
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

        # address cases where the tensor dims of the two imputs diverge
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

    def __build_block(self, action_L, action_R, filters, stride, inputs, adapt_depth: bool):
        '''
        Generate a block, following PNAS conventions.

        Args:
            action_L (tuple<int, string>): [description]
            action_R (tuple<int, string>): [description]
            filters (int): [description]
            stride (tuple<int, int>): [description]
            inputs (list<tf.keras.layers.Layer>): [description]
            adapt_depth (bool): [description]

        Returns:
            (tf.tensor): Output of Add keras layer
        '''
        input_index_L, action_name_L = action_L
        input_index_R, action_name_R = action_R

        # in reduction cell, still use stride (1, 1) if not using "original inputs" (-1, -2, no reduction for other blocks' outputs)
        stride_L = stride if input_index_L < 0 else (1, 1)
        stride_R = stride if input_index_R < 0 else (1, 1)

        # parse_action returns a custom layer model, that is then called with chosen input
        left_layer = self.__parse_action(filters, action_name_L, adapt_depth, strides=stride_L, tag='L')(inputs[input_index_L])
        right_layer = self.__parse_action(filters, action_name_R, adapt_depth, strides=stride_R, tag='R')(inputs[input_index_R])
        return layers.Add()([left_layer, right_layer])

    def __parse_action(self, filters, action, adapt_depth: bool, strides=(1, 1), tag='L'):
        '''
        Generate correct custom layer for provided action. Certain cases are handled incorrectly,
        so that model can still be built, albeit not with original specification

        # Args:
            filters: number of filters
            action: action string
            adapt_depth (bool): adapt depth of operations that don't alter it
            strides: stride to reduce spatial size
            tag (string): either L or R, identifing the block operation

        # Returns:
            (tf.keras.Model): The custom layer corresponding to the action (see ops.py)
        '''

        # check non parametrized operations first since they don't require a regex and are faster
        if action == 'identity':
            # 'identity' action case, if using (2, 2) stride it's actually handled as a pointwise convolution
            if strides == (2, 2):
                model_name = f'pointwise_conv_c{self.cell_index}b{self.block_index}{tag}'
                x = ops.Convolution(filters, kernel=(1, 1), strides=strides, name=model_name, weight_norm=self.weight_norm)
                return x
            else:
                # else just submits a linear layer if shapes match
                model_name = f'identity_c{self.cell_index}b{self.block_index}{tag}'
                x = ops.Identity(filters, strides, name=model_name)
                return x

        # check for separable conv
        match = self.op_regexes['dconv'].match(action)  # type: re.Match
        if match:
            model_name = f'{match.group(1)}x{match.group(2)}_dconv_c{self.cell_index}b{self.block_index}{tag}'
            x = ops.SeperableConvolution(filters, kernel=to_int_tuple(match.group(1, 2)), strides=strides,
                                         name=model_name, weight_norm=self.weight_norm)
            return x

        # check for stacked conv operation
        match = self.op_regexes['stack_conv'].match(action)  # type: re.Match
        if match:
            f = [filters, filters]
            k = [to_int_tuple(match.group(1, 2)), to_int_tuple(match.group(3, 4))]
            s = [strides, (1, 1)]

            model_name = f'{match.group(1)}x{match.group(2)}-{match.group(3)}x{match.group(4)}_conv_c{self.cell_index}b{self.block_index}{tag}'
            x = ops.StackedConvolution(f, k, s, name=model_name, weight_norm=self.weight_norm)
            return x

        # check for standard conv
        match = self.op_regexes['conv'].match(action)  # type: re.Match
        if match:
            model_name = f'3x3_conv_c{self.cell_index}b{self.block_index}{tag}'
            x = ops.Convolution(filters, kernel=to_int_tuple(match.group(1, 2)), strides=strides,
                                name=model_name, weight_norm=self.weight_norm)
            return x

        # check for pooling
        match = self.op_regexes['pool'].match(action)  # type: re.Match
        if match:
            size = to_int_tuple(match.group(1, 2))
            pool_type = match.group(3)

            model_name = f'{match.group(1)}x{match.group(2)}_{pool_type}pool_c{self.cell_index}b{self.block_index}{tag}'
            x = ops.PoolingConv(filters, pool_type, size, strides, name=model_name, weight_norm=self.weight_norm) if adapt_depth \
                else ops.Pooling(pool_type, size, strides, name=model_name)

            return x

        raise ValueError('Operation not covered by POPNAS algorithm')

    def define_callbacks(self, tb_logdir):
        '''
        Define callbacks used in model training.

        Returns:
            (tf.keras.Callback[]): Keras callbacks
        '''
        model_callbacks = []

        # TODO: Save best weights, not really necessary? Was used only to get best val_accuracy...
        # ckpt_callback = callbacks.ModelCheckpoint(filepath=log_service.build_path('temp_weights', 'cp_e{epoch:02d}_vl{val_accuracy:.2f}.ckpt'),
        #                                           save_weights_only=True, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)
        # model_callbacks.append(ckpt_callback)

        # By default shows losses and metrics for both training and validation
        tb_callback = callbacks.TensorBoard(log_dir=tb_logdir, profile_batch=0, histogram_freq=0, update_freq='epoch')
        model_callbacks.append(tb_callback)

        # TODO: convert into a parameter
        early_stop = False
        if early_stop:
            es_callback = callbacks.EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1)
            model_callbacks.append(es_callback)

        return model_callbacks

    def define_training_hyperparams_and_metrics(self, lr=0.01):
        loss = losses.CategoricalCrossentropy()
        optimizer = optimizers.Adam(learning_rate=lr)
        metrics = ['accuracy']

        # TODO: pnas used cosine decay with SGD, instead of Adam. Investigate which alternative is better

        return loss, optimizer, metrics
