import operator
import random
import re
from abc import abstractmethod, ABC

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, SeparableConv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, \
    Conv1D, Conv1DTranspose, SeparableConv1D, MaxPooling1D, AveragePooling1D, GlobalAveragePooling1D, \
    BatchNormalization, Layer
from tensorflow.keras.regularizers import Regularizer

from utils.func_utils import to_int_tuple
import operator
import random
import re
from abc import abstractmethod, ABC

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, SeparableConv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, \
    Conv1D, Conv1DTranspose, SeparableConv1D, MaxPooling1D, AveragePooling1D, GlobalAveragePooling1D, \
    BatchNormalization, Layer
from tensorflow.keras.regularizers import Regularizer

from utils.func_utils import to_int_tuple

# TODO: it could be nice to import only the correct ones and rename them so that are recognized by layers,
#  but input dims are given at runtime, making it difficult (inner classes with delayed import seems not possible)
op_dim_selector = {
    'conv': {1: Conv1D, 2: Conv2D},
    'tconv': {1: Conv1DTranspose, 2: Conv2DTranspose},
    'dconv': {1: SeparableConv1D, 2: SeparableConv2D},
    'max_pool': {1: MaxPooling1D, 2: MaxPooling2D},
    'avg_pool': {1: AveragePooling1D, 2: AveragePooling2D},
    'gap': {1: GlobalAveragePooling1D, 2: GlobalAveragePooling2D}
}


class OpInstantiator:
    '''
    Class that takes care of building and returning valid Keras layers for the operators and input shape considered.
    Based on input shape, 1D or 2D operators are used.
    '''
    def __init__(self, input_dims: int, reduction_stride_factor: int = 2, weight_reg: Regularizer = None):
        self.weight_reg = weight_reg

        self.op_dims = input_dims - 1
        self.reduction_stride = tuple([reduction_stride_factor] * self.op_dims)
        self.normal_stride = tuple([1] * self.op_dims)

        self.gap = op_dim_selector['gap'][self.op_dims]

        self.op_regexes = self.__compile_op_regexes()

    def __compile_op_regexes(self):
        '''
        Build a dictionary with compiled regexes for each parametrized supported operation.
        Adapt regexes based on input dimensionality.

        Returns:
            (dict): Regex dictionary
        '''
        # add groups to detect kernel size, based on op dimensionality.
        # e.g. Conv2D -> 3x3 conv, Conv1D -> 3 conv
        op_kernel_groups = 'x'.join([r'(\d+)'] * self.op_dims)

        return {'conv': re.compile(rf'{op_kernel_groups} conv'),
                'dconv': re.compile(rf'{op_kernel_groups} dconv'),
                'tconv': re.compile(rf'{op_kernel_groups} tconv'),
                'stack_conv': re.compile(rf'{op_kernel_groups}-{op_kernel_groups} conv'),
                'pool': re.compile(rf'{op_kernel_groups} (max|avg)pool')}

    def generate_pointwise_conv(self, filters: int, strided: bool, name: str):
        '''
        Provide builder for generating a pointwise convolution easily, for tensor shape regularization purposes.
        '''
        strides = self.reduction_stride if strided else self.normal_stride
        return Convolution(filters, tuple([1] * self.op_dims), strides=strides, name=name)

    def build_op_layer(self, op_name: str, filters: int, input_filters: int, layer_name_suffix: str, strided: bool = False) -> Layer:
        '''
        Generate a custom Keras layer for the provided operator and parameter. Certain operations are handled in a different way
        when used in reduction cells, compared to the normal cells, to handle the tensor shape changes and allow addition at the end of a block.

        # Args:
            op_name: operator to use
            filters: number of filters
            input_filters: number of filters of the input of this new layer
            layer_name_suffix: suffix appended to layer name, basically unique metadata about block and cell indexes
            strided: if op must use a stride different from 1 (reduction cells)

        # Returns:
            (tf.keras.layers.Layer): The custom layer corresponding to the operator
        '''
        adapt_depth = filters != input_filters
        strides = self.reduction_stride if strided else self.normal_stride

        # check non parametrized operations first since they don't require a regex and are faster
        if op_name == 'identity':
            # 'identity' action case, if using stride-2, then it's actually handled as a pointwise convolution
            if strided or adapt_depth:
                # TODO: IdentityReshaper leads to a strange non-deterministic bug and for now it has been disabled, reverting to pointwise convolution
                # layer_name = f'identity_reshaper{block_info_suffix}'
                # x = IdentityReshaper(filters, input_filters, strides, name=layer_name)
                layer_name = f'pointwise_id{layer_name_suffix}'
                kernel = tuple([1] * self.op_dims)
                return Convolution(filters, kernel, strides, weight_reg=self.weight_reg, name=layer_name)
            else:
                # else just submits a linear layer if shapes match
                layer_name = f'identity{layer_name_suffix}'
                return Identity(name=layer_name)

        # check for separable conv
        match = self.op_regexes['dconv'].match(op_name)  # type: re.Match
        if match:
            layer_name = f'{"x".join(match.groups())}_dconv{layer_name_suffix}'
            return SeparableConvolution(filters, kernel=to_int_tuple(match.groups()), strides=strides,
                                        name=layer_name, weight_reg=self.weight_reg)

        # check for transpose conv
        match = self.op_regexes['tconv'].match(op_name)  # type: re.Match
        if match:
            layer_name = f'{"x".join(match.groups())}_tconv{layer_name_suffix}'
            return TransposeConvolutionStack(filters, kernel=to_int_tuple(match.groups()), strides=strides,
                                             name=layer_name, weight_reg=self.weight_reg)

        # check for stacked conv operation
        match = self.op_regexes['stack_conv'].match(op_name)  # type: re.Match
        if match:
            g_count = len(match.groups())
            first_kernel_groups = match.groups()[:g_count // 2]
            second_kernel_groups = match.groups()[g_count // 2:]

            f = [filters, filters]
            k = [to_int_tuple(first_kernel_groups), to_int_tuple(second_kernel_groups)]
            s = [strides, self.normal_stride]

            layer_name = f'{"x".join(first_kernel_groups)}-{"x".join(second_kernel_groups)}_conv{layer_name_suffix}'
            return StackedConvolution(f, k, s, name=layer_name, weight_reg=self.weight_reg)

        # check for standard conv
        match = self.op_regexes['conv'].match(op_name)  # type: re.Match
        if match:
            layer_name = f'{"x".join(match.groups())}_conv{layer_name_suffix}'
            return Convolution(filters, kernel=to_int_tuple(match.groups()), strides=strides,
                               name=layer_name, weight_reg=self.weight_reg)

        # check for pooling
        match = self.op_regexes['pool'].match(op_name)  # type: re.Match
        if match:
            # last group is the pooling type
            pool_size = match.groups()[:-1]
            pool_type = match.groups()[-1]

            layer_name = f'{"x".join(pool_size)}_{pool_type}pool{layer_name_suffix}'
            return PoolingConv(filters, pool_type, to_int_tuple(pool_size), strides, name=layer_name, weight_reg=self.weight_reg) if adapt_depth \
                else Pooling(pool_type, to_int_tuple(pool_size), strides, name=layer_name)

        raise ValueError(f'Operator malformed or not covered by POPNAS algorithm: {op_name}')


class Identity(Layer):
    def __init__(self, name='identity', **kwargs):
        '''
        Simply adds an identity connection.
        '''
        super().__init__(name=name, **kwargs)

    def call(self, inputs, training=None, mask=None):
        return tf.identity(inputs)


# TODO: deprecated in previous versions, even if it should work fine for 2D. Right now it needs a refactor to adapt to different input dims.
# class IdentityReshaper(Layer):
#     def __init__(self, filters, input_filters, strides, name='identity_reshaper', **kwargs):
#         '''
#         Identity alternative when the tensor shape between input and output differs.
#         IdentityReshaper can apply a stride without doing any operation,
#         also adapting depth by replicating multiple times the tensor and concatenating the replicas on depth axis.
#         '''
#         super().__init__(name=name, **kwargs)
#         self.filters = filters
#         self.input_filters = input_filters
#         self.strides = strides
#         self.no_stride = self.strides[0] == 1 and self.strides[1] == 1
#
#         self.replication_factor = math.ceil(filters / input_filters)
#
#     def call(self, inputs, training=None, mask=None):
#         # TODO: it seems that ::1 slice is bugged, if stride is used when not needed the model has random output. TF bug?
#         input_stride = inputs if self.no_stride else inputs[:, ::self.strides[0], ::self.strides[1], :]
#         return tf.tile(input_stride, [1, 1, 1, self.replication_factor])[:, :, :, :self.filters]
#
#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             'filters': self.filters,
#             'input_filters': self.input_filters,
#             'strides': self.strides
#         })
#         return config


class OpBatchActivation(Layer, ABC):
    @abstractmethod
    def __init__(self, filters, kernel, strides, weight_reg=None, name='abstract', **kwargs):
        '''
        Abstract utility class used as baseline for any {Operation - Batch Normalization - Activation} layer.
        Op attribute must be set to a Keras layer or TF nn operation in all concrete implementations.
        '''
        super().__init__(name=name, **kwargs)
        self.filters = filters
        self.kernel = kernel
        self.strides = strides
        self.weight_reg = weight_reg

        self.bn = BatchNormalization()

        # concrete implementations must use a valid Keras layer / TF operation, assigning it to this variable during __init__
        self.op = None

    def call(self, inputs, training=None, mask=None):
        x = self.op(inputs)
        x = self.bn(x, training=training)
        return tf.nn.silu(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel': self.kernel,
            'strides': self.strides,
            'weight_reg': self.weight_reg
        })
        return config


class SeparableConvolution(OpBatchActivation):
    def __init__(self, filters, kernel, strides, weight_reg=None, name='dconv', **kwargs):
        '''
        Constructs a {Separable Convolution - Batch Normalization - Activation} layer.
        '''
        super().__init__(filters, kernel, strides, weight_reg, name=name, **kwargs)

        dconv = op_dim_selector['dconv'][len(kernel)]
        self.op = dconv(filters, kernel, strides=strides, padding='same',
                        depthwise_initializer='he_uniform', pointwise_initializer='he_uniform',
                        depthwise_regularizer=weight_reg, pointwise_regularizer=weight_reg)


class Convolution(OpBatchActivation):
    def __init__(self, filters, kernel, strides, weight_reg=None, name='conv', **kwargs):
        '''
        Constructs a {Spatial Convolution - Batch Normalization - Activation} layer.
        '''
        super().__init__(filters, kernel, strides, weight_reg, name=name, **kwargs)

        conv = op_dim_selector['conv'][len(kernel)]
        self.op = conv(filters, kernel, strides=strides, padding='same',
                         kernel_initializer='he_uniform', kernel_regularizer=weight_reg)


class TransposeConvolution(OpBatchActivation):
    def __init__(self, filters, kernel, strides, weight_reg=None, name='tconv', **kwargs):
        '''
        Constructs a {Transpose Convolution - Batch Normalization - Activation} layer.
        '''
        super().__init__(filters, kernel, strides, weight_reg, name=name, **kwargs)

        tconv = op_dim_selector['tconv'][len(kernel)]
        self.op = tconv(filters, kernel, strides, padding='same',
                                  kernel_initializer='he_uniform', kernel_regularizer=weight_reg)


class TransposeConvolutionStack(Layer):
    def __init__(self, filters, kernel, strides, weight_reg=None, name='tconv', **kwargs):
        '''
        Constructs a Transpose Convolution - Convolution layer. Batch Normalization and Relu are applied on both.
        '''
        super().__init__(name=name, **kwargs)
        self.filters = filters
        self.kernel = kernel
        self.strides = strides
        self.weight_reg = weight_reg

        transpose_stride = tuple([2] * len(strides))
        conv_strides = tuple(map(operator.mul, transpose_stride, strides))

        self.transposeConv = TransposeConvolution(filters, kernel, transpose_stride, weight_reg)
        self.conv = Convolution(filters, kernel, conv_strides, weight_reg)

    def call(self, inputs, training=None, mask=None):
        x = self.transposeConv(inputs)
        return self.conv(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel': self.kernel,
            'strides': self.strides,
            'weight_reg': self.weight_reg
        })
        return config


class StackedConvolution(Layer):
    def __init__(self, filter_list, kernel_list, stride_list, weight_reg=None, name='stack_conv', **kwargs):
        '''
        Constructs a stack of Convolution blocks that are chained together.
        '''
        super().__init__(name=name, **kwargs)
        self.filter_list = filter_list
        self.kernel_list = kernel_list
        self.stride_list = stride_list
        self.weight_reg = weight_reg

        assert len(filter_list) == len(kernel_list) and len(kernel_list) == len(stride_list), "List lengths must match"

        self.convs = [Convolution(f, k, s, weight_reg=weight_reg) for (f, k, s) in zip(filter_list, kernel_list, stride_list)]

    def call(self, inputs, training=None, mask=None):
        x = inputs

        for conv in self.convs:
            x = conv(x, training=training)

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'filter_list': self.filter_list,
            'kernel_list': self.kernel_list,
            'stride_list': self.stride_list,
            'weight_reg': self.weight_reg
        })
        return config


class Pooling(Layer):
    def __init__(self, pool_type, size, strides, name='pool', **kwargs):
        '''
        Constructs a standard pooling layer (average or max).
        '''
        super().__init__(name=name, **kwargs)
        self.pool_type = pool_type
        self.size = size
        self.strides = strides

        pool_layer = op_dim_selector[f'{pool_type}_pool'][len(size)]
        self.pool = pool_layer(size, strides, padding='same')

    def call(self, inputs, training=None, mask=None):
        return self.pool(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({
            'pool_type': self.pool_type,
            'size': self.size,
            'strides': self.strides
        })
        return config


class PoolingConv(Layer):
    def __init__(self, filters, pool_type, size, strides, weight_reg=None, name='pool_conv', **kwargs):
        '''
        Constructs a pooling layer (average or max). It also adds a pointwise convolution (using Convolution class)
        to adapt the output depth to filters size.
        '''
        super().__init__(name=name, **kwargs)
        self.filters = filters
        self.pool_type = pool_type
        self.size = size
        self.strides = strides
        self.weight_reg = weight_reg

        self.pool = Pooling(pool_type, size, strides)

        ones = tuple([1] * len(size))
        self.pointwise_conv = Convolution(filters, kernel=ones, strides=ones, weight_reg=weight_reg)

    def call(self, inputs, training=None, mask=None):
        output = self.pool(inputs)
        return self.pointwise_conv(output)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'pool_type': self.pool_type,
            'size': self.size,
            'strides': self.strides,
            'weight_reg': self.weight_reg
        })
        return config


# TODO: probably must be adapted for different input sizes (was intended for 2D)
# TODO: this scheduled drop path implementation is inspired by the one implemented in Tensorflow for PNASNetV5 in these links:
#  https://github.com/chenxi116/PNASNet.TF/blob/338371ffc3122498dc71aff9d59001f40ef22e6c/cell.py#L136
#  https://github.com/tensorflow/models/blob/30e6e03f66efad4e43f1b98ec8680451f5a86a72/research/slim/nets/nasnet/nasnet_utils.py#L432
#  but these versions actually differs from the one used in FractalNet, since they don't check that at least one path survives!
#  This leads to awful accuracies also on training set, since a cell/block could output a totally 0-ed tensor.
#  Refactored to work on a list of tensors, before "join" layers (add, concatenate). Seems to work correctly now, but it is written for easier debug
#  and it's surely inefficient right now (adds about 12 seconds of overhead to training time).
#  Investigate on better usage of TF API and rewrite it with performance in mind.
class ScheduledDropPath(Layer):
    def __init__(self, keep_probability: float, cell_ratio: float, total_training_steps: int, name='scheduled_drop_path', **kwargs):
        super().__init__(name=name, **kwargs)
        self.keep_probability = keep_probability
        self.cell_ratio = cell_ratio  # (self._cell_num + 1) / float(self._total_num_cells)
        self.total_training_steps = total_training_steps  # number of times weights are updated (batches_per_epoch * epochs)
        self.current_step = tf.Variable(0, trainable=False, dtype=tf.float32)

    def call(self, inputs, training=None, mask=None):
        if training and self.keep_probability < 1.0:
            # Scale keep prob by cell number
            keep_prob = 1 - self.cell_ratio * (1 - self.keep_probability)

            # Decrease keep prob over time (global_step is the current batch number)
            # current_step = tf.cast(tf.compat.v1.train.get_or_create_global_step(), tf.float32)
            current_ratio = self.current_step / self.total_training_steps
            keep_prob = 1 - current_ratio * (1 - keep_prob)

            # Drop path
            # noise_shape = [tf.shape(input=inputs)[0], 1, 1, 1]
            # random_tensor = keep_prob
            # random_tensor += tf.random.uniform(noise_shape, dtype=tf.float32)
            # binary_tensor = tf.cast(tf.floor(random_tensor), inputs.dtype)

            noise_shape = [tf.shape(input=inputs[0])[0], 1, 1, 1]
            input_dtype = inputs[0].dtype
            binary_tensors = []
            for i in range(len(inputs)):
                random_tensor = keep_prob
                random_tensor += tf.random.uniform(noise_shape, dtype=tf.float32)
                binary_tensors.append(tf.cast(tf.floor(random_tensor), dtype=input_dtype))

            agg_mask_sum = tf.math.add_n(binary_tensors)
            ensure_path_tensor = tf.maximum(0.0, 1 - agg_mask_sum)
            # mask_random_index = tf.random.uniform(shape=[], maxval=len(inputs), dtype=tf.int32)
            mask_random_index = random.randrange(len(inputs))

            binary_tensors[mask_random_index] = tf.add(binary_tensors[mask_random_index], ensure_path_tensor)

            keep_prob_inv = tf.cast(1.0 / keep_prob, dtype=input_dtype)
            self.current_step.assign_add(delta=1)

            output_tensors = []
            for i in range(len(inputs)):
                output_tensors.append(tf.multiply(tf.multiply(inputs[i], keep_prob_inv), binary_tensors[i]))

            # return inputs * keep_prob_inv * binary_tensor
            return output_tensors
        else:
            return inputs

    def get_config(self):
        config = super().get_config()
        config.update({
            'keep_probability': self.keep_probability,
            'cell_ratio': self.cell_ratio,
            'total_training_steps': self.total_training_steps
        })
        return config
