import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Conv2D, SeparableConv2D, MaxPooling2D, AveragePooling2D, BatchNormalization

def pad_closure(desired_depth):
    '''
    Pad depth of an identity tensor with zeros if has not already the right depth for performing addition at the end of the block.

    Args:
        desired_depth (integer): Depth required for addition with other tensor inside the block
    '''
    def pad_identity(inputs):
        input_depth = inputs.get_shape().as_list()[3]
        pad_size = desired_depth - input_depth

        if pad_size > 0:
            paddings = tf.constant([[0, 0], [0, 0], [0, 0], [0, pad_size]])
            return tf.pad(inputs, paddings)  # constant, with 0s
        else:
            return inputs

    return pad_identity

class Identity(Model):

    def __init__(self, filters, strides):
        '''
        Simply adds an identity connection, padding in depth with 0 if necessary to enable block add operator.
        '''
        super(Identity, self).__init__()

        #self.op = lambda x : x
        #self.op = Layer()   # identity
        self.op = pad_closure(filters)

    def call(self, inputs, training=None, mask=None):
        return self.op(inputs)


class SeperableConvolution(Model):

    def __init__(self, filters, kernel, strides):
        '''
        Constructs a Seperable Convolution - Batch Normalization - Relu block.
        '''
        super(SeperableConvolution, self).__init__()

        self.conv = SeparableConv2D(filters, kernel, strides=strides, padding='same',
                                    kernel_initializer='he_uniform')
        self.bn = BatchNormalization()

    def call(self, inputs, training=None, mask=None):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        return tf.nn.relu(x)


class Convolution(Model):

    def __init__(self, filters, kernel, strides):
        '''
        Constructs a Spatial Convolution - Batch Normalization - Relu block.
        '''
        super(Convolution, self).__init__()

        self.conv = Conv2D(filters, kernel, strides=strides, padding='same',
                           kernel_initializer='he_uniform')
        self.bn = BatchNormalization()

    def call(self, inputs, training=None, mask=None):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        return tf.nn.relu(x)


class StackedConvolution(Model):

    def __init__(self, filter_list, kernel_list, stride_list):
        '''
        Constructs a stack of Convolution blocks that are chained together.
        '''
        super(StackedConvolution, self).__init__()

        assert len(filter_list) == len(kernel_list) and len(kernel_list) == len(stride_list), "List lengths must match"

        self.convs = []
        for i, (f, k, s) in enumerate(zip(filter_list, kernel_list, stride_list)):
            conv = Convolution(f, k, s)
            self.convs.append(conv)

    def call(self, inputs, training=None, mask=None):
        x = inputs

        for conv in self.convs:
            x = conv(x, training=training)

        return x


class Pooling(Model):

    def __init__(self, type, size, strides):
        '''
        Constructs a pooling layer (average or max).
        '''
        super(Pooling, self).__init__()
        if type == 'max':
            self.pool = MaxPooling2D(size, strides, padding='same')
        else:
            self.pool = AveragePooling2D(size, strides, padding='same')
    def call(self, inputs, training=None, mask=None):
        return self.pool(inputs)
