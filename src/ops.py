import tensorflow as tf
from tensorflow.keras.layers import Conv2D, SeparableConv2D, MaxPooling2D, AveragePooling2D, BatchNormalization, Layer


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
    def __init__(self, filters, name='identity', **kwargs):
        '''
        Simply adds an identity connection, padding in depth with 0 if necessary to enable block add operator.
        '''
        super().__init__(name=name, **kwargs)
        self.filters = filters

        self.op = Layer()  # Identity layer in Keras
        self.identity_call = depth_zero_pad_closure(filters, self.op)

    def call(self, inputs, training=None, mask=None):
        return self.identity_call(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({'filters': self.filters})
        return config


class SeperableConvolution(Layer):
    def __init__(self, filters, kernel, strides, weight_norm=None, name='dconv', **kwargs):
        '''
        Constructs a Seperable Convolution - Batch Normalization - Relu block.
        '''
        super().__init__(name=name, **kwargs)
        self.filters = filters
        self.kernel = kernel
        self.strides = strides
        self.weight_norm = weight_norm

        self.conv = SeparableConv2D(filters, kernel, strides=strides, padding='same',
                                    depthwise_initializer='he_uniform', pointwise_initializer='he_uniform',
                                    depthwise_regularizer=weight_norm, pointwise_regularizer=weight_norm)
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
            'weight_norm': self.weight_norm
        })
        return config


class Convolution(Layer):
    def __init__(self, filters, kernel, strides, weight_norm=None, name='conv', **kwargs):
        '''
        Constructs a Spatial Convolution - Batch Normalization - Relu block.
        '''
        super().__init__(name=name, **kwargs)
        self.filters = filters
        self.kernel = kernel
        self.strides = strides
        self.weight_norm = weight_norm

        self.conv = Conv2D(filters, kernel, strides=strides, padding='same',
                           kernel_initializer='he_uniform', kernel_regularizer=weight_norm)
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
            'weight_norm': self.weight_norm
        })
        return config


class StackedConvolution(Layer):
    def __init__(self, filter_list, kernel_list, stride_list, weight_norm=None, name='stack_conv', **kwargs):
        '''
        Constructs a stack of Convolution blocks that are chained together.
        '''
        super().__init__(name=name, **kwargs)
        self.filter_list = filter_list
        self.kernel_list = kernel_list
        self.stride_list = stride_list
        self.weight_norm = weight_norm

        assert len(filter_list) == len(kernel_list) and len(kernel_list) == len(stride_list), "List lengths must match"

        self.convs = []
        for (f, k, s) in zip(filter_list, kernel_list, stride_list):
            conv = Convolution(f, k, s, weight_norm=weight_norm)
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
            'weight_norm': self.weight_norm
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
    def __init__(self, filters, pool_type, size, strides, weight_norm=None, name='pool_conv', **kwargs):
        '''
        Constructs a pooling layer (average or max). It also adds a pointwise convolution (using Convolution class)
        to adapt the output depth to filters size.
        '''
        super().__init__(name=name, **kwargs)
        self.filters = filters
        self.pool_type = pool_type
        self.size = size
        self.strides = strides
        self.weight_norm = weight_norm

        self.pool = Pooling(pool_type, size, strides)
        self.pointwise_conv = Convolution(filters, kernel=(1, 1), strides=(1, 1), weight_norm=weight_norm)

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
            'weight_norm': self.weight_norm
        })
        return config


# TODO: this scheduled drop path implementation is heavily inspired by the one implemented in Tensorflow for PNASNetV5 in these links:
#  https://github.com/chenxi116/PNASNet.TF/blob/338371ffc3122498dc71aff9d59001f40ef22e6c/cell.py#L136
#  https://github.com/tensorflow/models/blob/30e6e03f66efad4e43f1b98ec8680451f5a86a72/research/slim/nets/nasnet/nasnet_utils.py#L432
#  but these versions actually differs from the one used in FractalNet, since they don't check that at least one path survives!
#  Should be reworked to take an array of tensors and make sure one survives.
class ScheduledDropPath(Layer):
    def __init__(self, keep_probability: float, cell_ratio: float, total_training_steps: int, name='scheduled_drop_path', **kwargs):
        super().__init__(name=name, **kwargs)
        self.keep_probability = keep_probability
        self.cell_ratio = cell_ratio    # (self._cell_num + 1) / float(self._total_num_cells)
        self.total_training_steps = total_training_steps    # number of times weights are updated (batches_per_epoch * epochs)
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
            noise_shape = [tf.shape(input=inputs)[0], 1, 1, 1]
            random_tensor = keep_prob
            random_tensor += tf.random.uniform(noise_shape, dtype=tf.float32)
            binary_tensor = tf.cast(tf.floor(random_tensor), inputs.dtype)
            keep_prob_inv = tf.cast(1.0 / keep_prob, inputs.dtype)

            self.current_step.assign_add(delta=1)
            return inputs * keep_prob_inv * binary_tensor
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
