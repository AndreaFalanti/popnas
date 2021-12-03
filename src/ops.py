import math
import operator
import random

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, SeparableConv2D, MaxPooling2D, AveragePooling2D, BatchNormalization, Layer


# TODO: actually not used, delete it?
def depth_zero_pad_closure(desired_depth, op_layer):
    '''
    Pad depth of a Keras layer with zeros, if it has not already the right depth for performing addition at the end of the block.

    Args:
        desired_depth (int): Depth required for addition with other tensor inside the block (filters value).
        op_layer (tf.keras.Layer): Keras layer to use. Only layers that don't modify depth can be used, otherwise the closure will fail
            (as it wouldn't be necessary to adapt the depth).

    Returns:
        (Callable): call function built by closure
    '''

    def depth_zero_pad_call(inputs):
        # directly compute depth on input, because 
        input_depth = inputs.get_shape().as_list()[3]
        pad_size = desired_depth - input_depth

        if pad_size > 0:
            output = op_layer(inputs)
            paddings = tf.constant([[0, 0], [0, 0], [0, 0], [0, pad_size]])
            return tf.pad(output, paddings)  # pad output with 0s
        else:
            return op_layer(inputs)

    return depth_zero_pad_call


# TODO: Convolution is spawned during call, which is not optimal for TF. For now it is abandoned, now pooling
#  use the check directly in code and pre-instantiate the convolution
def depth_pointwise_conv_closure(desired_depth, op_layer):
    '''
    Generate call for pooling operation. It adds a pointwise convolution to adapt the depth, if necessary.

    Args:
        desired_depth (int): Depth required for addition with other tensor inside the block (filters value).
        op_layer (tf.keras.Layer): Keras layer to use. Only layers that don't modify depth can be used, otherwise the closure will fail
            (as it wouldn't be necessary to adapt the depth).

    Returns:
        (Callable): call function built by closure
    '''

    def depth_pointwise_conv_call(inputs):
        input_depth = inputs.get_shape().as_list()[3]

        if input_depth != desired_depth:
            output = op_layer(inputs)
            return Convolution(desired_depth, (1, 1), (1, 1))(output)
        else:
            return op_layer(inputs)

    return depth_pointwise_conv_call


class Identity(Layer):
    def __init__(self, name='identity', **kwargs):
        '''
        Simply adds an identity connection.
        '''
        super().__init__(name=name, **kwargs)

    def call(self, inputs, training=None, mask=None):
        return tf.identity(inputs)


class IdentityReshaper(Layer):
    def __init__(self, filters, input_filters, strides, name='identity_reshaper', **kwargs):
        '''
        Identity alternative when the tensor shape between input and output differs.
        IdentityReshaper can apply a stride without doing any operation,
        also adapting depth by replicating multiple times the tensor and concatenating the replicas on depth axis.
        '''
        super().__init__(name=name, **kwargs)
        self.filters = filters
        self.input_filters = input_filters
        self.strides = strides

        self.replication_factor = math.ceil(filters / input_filters)

    def call(self, inputs, training=None, mask=None):
        input_stride = inputs[::1, ::self.strides[0], ::self.strides[1], ::1]
        return tf.tile(input_stride, [1, 1, 1, self.replication_factor])[:, :, :, :self.filters]

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'strides': self.strides
        })
        return config


class SeparableConvolution(Layer):
    def __init__(self, filters, kernel, strides, weight_reg=None, name='dconv', **kwargs):
        '''
        Constructs a Separable Convolution - Batch Normalization - Relu block.
        '''
        super().__init__(name=name, **kwargs)
        self.filters = filters
        self.kernel = kernel
        self.strides = strides
        self.weight_reg = weight_reg

        self.conv = SeparableConv2D(filters, kernel, strides=strides, padding='same',
                                    depthwise_initializer='he_uniform', pointwise_initializer='he_uniform',
                                    depthwise_regularizer=weight_reg, pointwise_regularizer=weight_reg)
        self.bn = BatchNormalization()

    def call(self, inputs, training=None, mask=None):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        return tf.nn.relu(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel': self.kernel,
            'strides': self.strides,
            'weight_reg': self.weight_reg
        })
        return config


# TODO: remove duplication between layers using (single op - bn - activation) by implementing an abstract Layer subclass
class Convolution(Layer):
    def __init__(self, filters, kernel, strides, weight_reg=None, name='conv', **kwargs):
        '''
        Constructs a Spatial Convolution - Batch Normalization - Relu block.
        '''
        super().__init__(name=name, **kwargs)
        self.filters = filters
        self.kernel = kernel
        self.strides = strides
        self.weight_reg = weight_reg

        self.conv = Conv2D(filters, kernel, strides=strides, padding='same',
                           kernel_initializer='he_uniform', kernel_regularizer=weight_reg)
        self.bn = BatchNormalization()

    def call(self, inputs, training=None, mask=None):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        return tf.nn.relu(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel': self.kernel,
            'strides': self.strides,
            'weight_reg': self.weight_reg
        })
        return config


class TransposeConvolution(Layer):
    def __init__(self, filters, kernel, strides, weight_reg=None, name='tconv', **kwargs):
        '''
        Constructs a Transpose Convolution - Convolution layer. Batch Normalization and Relu are applied on both.
        '''
        super().__init__(name=name, **kwargs)
        self.filters = filters
        self.kernel = kernel
        self.strides = strides
        self.weight_reg = weight_reg

        transpose_stride = (2, 2)
        conv_strides = tuple(map(operator.mul, transpose_stride, strides))

        self.transposeConv = Conv2DTranspose(filters, kernel, transpose_stride, padding='same',
                                             kernel_initializer='he_uniform', kernel_regularizer=weight_reg)
        self.conv = Conv2D(filters, kernel, conv_strides, padding='same', kernel_initializer='he_uniform', kernel_regularizer=weight_reg)
        self.bn = BatchNormalization()
        self.bn2 = BatchNormalization()

    def call(self, inputs, training=None, mask=None):
        x = self.transposeConv(inputs)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv(x)
        x = self.bn2(x, training=training)
        return tf.nn.relu(x)

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

        self.convs = []
        for (f, k, s) in zip(filter_list, kernel_list, stride_list):
            conv = Convolution(f, k, s, weight_reg=weight_reg)
            self.convs.append(conv)

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

        if pool_type == 'max':
            self.pool = MaxPooling2D(size, strides, padding='same')
        else:
            self.pool = AveragePooling2D(size, strides, padding='same')

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
        self.pointwise_conv = Convolution(filters, kernel=(1, 1), strides=(1, 1), weight_reg=weight_reg)

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

    # def build(self, input_shape):
    #     noise_shape = [input_shape[0][0], 1, 1, 1]
    #     self.binary_tensors = []
    #
    #     for i in range(len(input_shape)):
    #         self.binary_tensors.append(tf.zeros(noise_shape))

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
