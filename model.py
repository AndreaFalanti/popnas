import numpy as np

import tensorflow as tf
from tensorflow.python.keras.models import Model

from ops import Identity
from ops import Convolution
from ops import StackedConvolution
from ops import SeperableConvolution
from ops import Pooling
from ops import GlobalAveragePooling2D
from ops import Dense


class ModelGenerator(Model):

    def __init__(self, actions):
        '''
        Utility Model class to construct child models provided with an action list.

        # Args:
            actions: list of [input; action] pairs that define the cell.
        '''
        super(ModelGenerator, self).__init__()

        self.B = len(actions) // 4

        if len(actions) > 0:
            self.action_list = np.split(np.array(actions), len(actions) // 2)
            self.M = 3
            self.N = 2
        else:
            self.action_list = []
            self.M = 1
            self.N = 0

        self.global_counter = 0

        self.cells = []
        filters = 24

        # add (M - 1) times N normal cells and a reduction cell
        for i in range(self.M - 1):
            # add N times a normal cell
            for j in range(self.N):
                normal_cell = self.build_cell(self.B, self.action_list, filters=filters, stride=(1, 1))
                self.cells.append(normal_cell)
            # add 1 time a reduction cell
            filters = filters * 2
            reduction_cell = self.build_cell(self.B, self.action_list, filters=filters, stride=(2, 2))
            self.cells.append(reduction_cell)
        # add N time a normal cell
        for i in range(self.N):
            normal_cell = self.build_cell(self.B, self.action_list, filters=filters, stride=(1, 1))
            self.cells.append(normal_cell)

        self.gap = GlobalAveragePooling2D(name='Avg')
        self.logits = Dense(10, activation='softmax', name='Softmax')  # only logits

    def call(self, inputs, training=None, mask=None):

        x = inputs
        for cell in self.cells:
            cell_ops = []
            for op in cell:
                out = op(x, training=training)
                cell_ops.append(out)
            x = tf.concat(cell_ops, axis=-1)

        x = self.gap(x)
        out = self.logits(x)

        return out

    def build_cell(self, B, action_list, filters, stride):
        # if cell size is 1 block only
        if B == 1:
            left = self.parse_action(filters, action_list[0][1], strides=stride)
            right = self.parse_action(filters, action_list[1][1], strides=stride)
            return [left, right]

        # else concatenate all the intermediate blocks
        actions = []
        for i in range(B):
            left_action = self.parse_action(filters, action_list[i * 2][1], strides=stride)
            right_action = self.parse_action(filters, action_list[i * 2 + 1][1], strides=stride)

            actions.extend([left_action, right_action])

        return actions

    def parse_action(self, filters, action, strides=(1, 1)):
        '''
        Parses the input string as an action. Certain cases are handled incorrectly,
        so that model can still be built, albeit not with original specification

        # Args:
            ip: input tensor
            filters: number of filters
            action: action string
            strides: stride to reduce spatial size

        # Returns:
            a tensor with an action performed
        '''
        # applies a 3x3 separable conv
        if action == '3x3 dconv':
            x = SeperableConvolution(filters, (3, 3), strides)
            x._name = '3x3_dconv'
            return x

        # applies a 5x5 separable conv
        if action == '5x5 dconv':
            x = SeperableConvolution(filters, (5, 5), strides)
            x._name = '5x5_dconv'
            return x

        # applies a 7x7 separable conv
        if action == '7x7 dconv':
            x = SeperableConvolution(filters, (7, 7), strides)
            x._name = '7x7_dconv'
            return x

        # applies a 1x7 and then a 7x1 standard conv operation
        if action == '1x7-7x1 conv':
            f = [filters, filters]
            k = [(1, 7), (7, 1)]
            s = [strides, 1]

            x = StackedConvolution(f, k, s)
            x._name = '1x7-7x1_conv'
            return x

        # applies a 3x3 standard conv
        if action == '3x3 conv':
            x = Convolution(filters, (3, 3), strides)
            x._name = '3x3_conv'
            return x

        # applies a 3x3 maxpool
        if action == '3x3 maxpool':
            x = Pooling('max', (3, 3), strides=strides)
            x._name = '3x3_maxpool'
            return x

        # applies a 3x3 avgpool
        if action == '3x3 avgpool':
            x = Pooling('avg', (3, 3), strides=strides)
            x._name = '3x3_avgpool'
            return x

        # attempts a linear operation (if size matches) or a strided linear conv projection to reduce spatial depth
        if strides == (2, 2):
            # channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
            # input_filters = K.int_shape(ip)[channel_axis]
            #
            # assert input_filters == filters, "Must perform identity op, but incorrect number of filters provided as input"
            x = Convolution(filters, (1, 1), strides)
            return x
        else:
            # else just submits a linear layer if shapes match
            x = Identity(filters, strides)
            x._name = 'identity'
            return x
