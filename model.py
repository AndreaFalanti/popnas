import numpy as np

import tensorflow as tf
from tensorflow.python.keras.models import Model

import ops

from tensorflow.python.keras.layers import Dense, GlobalAveragePooling2D

# TODO: refactor this in a function, like the ones we have done in NN course. This will make this a lot more simple.
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
            # TODO: use tuples instead of list of lists?
            #self.action_list = np.split(np.array(actions), len(actions) // 2)
            self.action_list = [x for x in zip(*[iter(actions)]*2)]     # generate a list of tuples (pairs)
            self.M = 3
            self.N = 2
        else:
            self.action_list = []
            self.M = 1
            self.N = 0

        # TODO: never used?
        self.global_counter = 0

        # used for layers naming, defined in class to avoid to pass them across multiple functions
        self.cell_index = 0
        self.block_index = 0

        self.model = self.build_model(32)


    def build_model(self, filters=32):
        # reset cell index, otherwise would use last model value
        self.cell_index = 0

        # TODO: dimensions are unknown a priori, but could be inferred by dataset used
        model_input = tf.keras.layers.Input(shape=(None, None, 3))

        # define inputs usable by blocks
        # last_output will be the input image at start, while skip_output is the output of block previous to last one (lag 1, used in actions with input = -2)
        last_output = model_input
        skip_output = last_output

        # TODO: refactor this to avoid updating output here everytime  
        # add (M - 1) times N normal cells and a reduction cell
        for i in range(self.M - 1):
            # add N times a normal cell
            for j in range(self.N):
                normal_cell = self.build_cell(self.B, self.action_list, filters=filters, stride=(1,1), inputs=[skip_output, last_output])
                self.cell_index += 1
                skip_output = last_output
                last_output = normal_cell

            # add 1 time a reduction cell
            filters = filters * 2
            reduction_cell = self.build_cell(self.B, self.action_list, filters=filters, stride=(2,2), inputs=[skip_output, last_output])
            self.cell_index += 1
            # TODO: shape discrepancy doesn't allow to use skip connection after reduction, for now simply set skip_output to the reduction_cell
            skip_output = reduction_cell
            last_output = reduction_cell

        # add N time a normal cell
        for i in range(self.N):
            normal_cell = self.build_cell(self.B, self.action_list, filters=filters, stride=(1,1), inputs=[skip_output, last_output])
            self.cell_index += 1
            skip_output = last_output
            last_output = normal_cell
                
        gap = GlobalAveragePooling2D(name='GAP')(last_output)
        # TODO: other datasets have a different number of classes, should be a parameter (10 constant)
        output = Dense(10, activation='softmax', name='Softmax')(gap) # only logits

        return tf.keras.Model(inputs = model_input, outputs = output)


    def call(self, inputs, training=None, mask=None):
        return self.model(inputs, training=training)

    def build_cell(self, B, action_list, filters, stride, inputs):
        '''
        Generate cell from action list. Following PNAS paper, addition is used to combine block results.

        Args:
            B (int): Number of blocks in the cell
            action_list (list<tuple<int, string>): List of tuples of 2 elements -> (input, action_name). Input can be either -1 (last cell output) or -2 (skip connection).
                                                TODO: add 0 for last block and allow block nesting
            filters (int): Initial filters to use
            stride (tuple<int, int>): (1, 1) for normal cells, (2, 2) for reduction cells
            inputs (list<tf.keras.layers.Layer>): Possible layers to use as input (based on action_list index value)

        Returns:
            (tf.keras.layers.Add | tf.keras.layers.Concatenate): [description]
        '''

        # if cell size is 1 block only
        if B == 1:
            return self.build_block(action_list[0], action_list[1], filters, stride, inputs)

        # else concatenate all the intermediate blocks that compose the cell
        # TODO: actually doesn't handle nesting between blocks outputs, producing only flat cells...
        out_layers = []
        for i in range(B):
            self.block_index = i
            out_layers.append(self.build_block(action_list[i * 2], action_list[i * 2 + 1], filters, stride, inputs))

        # concatenate all 'Add' layers, outputs of each single block
        concat_layer = tf.keras.layers.Concatenate(axis=-1)(out_layers)
        # reduce depth to filters value, otherwise concatenation would lead to (b * filters) tensor depth
        x = ops.Convolution(filters, (1, 1), (1, 1))
        x._name = f'concat_pointwise_conv_c{self.cell_index}'
        return x(concat_layer)

    def build_block(self, action_L, action_R, filters, stride, inputs):
        '''
        Generate a block, following PNAS conventions.

        Args:
            action_L (tuple<int, string>): [description]
            action_R (tuple<int, string>): [description]
            filters (int): [description]
            stride (tuple<int, int>): [description]
            inputs (list<tf.keras.layers.Layer>): [description]

        Returns:
            (tf.keras.layers.Add): [description]
        '''
        input_index_L, action_name_L  = action_L
        input_index_R, action_name_R = action_R

        # parse_action returns a custom layer model, that is then called with chosen input
        left_layer = self.parse_action(filters, action_name_L, strides=stride, tag='L')(inputs[input_index_L])
        right_layer = self.parse_action(filters, action_name_R, strides=stride, tag='R')(inputs[input_index_R])
        return tf.keras.layers.Add()([left_layer, right_layer])

    def parse_action(self, filters, action, strides=(1, 1), tag='L'):
        '''
        Generate correct custom layer for provided action. Certain cases are handled incorrectly,
        so that model can still be built, albeit not with original specification

        # Args:
            filters: number of filters
            action: action string
            strides: stride to reduce spatial size
            tag (string): either L or R, identifing the block operation

        # Returns:
            (tf.keras.Model): The custom layer corresponding to the action (see ops.py)
        '''

        # basically a huge switch case, python has no switch case because 'reasons'...
        # TODO: with python 3.10.0 match-case is available, it's worth to upgrade python for it?

        # applies a 3x3 separable conv
        if action == '3x3 dconv':
            x = ops.SeperableConvolution(filters, (3, 3), strides)
            x._name = f'3x3_dconv_c{self.cell_index}b{self.block_index}{tag}'
            return x

        # applies a 5x5 separable conv
        if action == '5x5 dconv':
            x = ops.SeperableConvolution(filters, (5, 5), strides)
            x._name = f'5x5_dconv_c{self.cell_index}b{self.block_index}{tag}'
            return x

        # applies a 7x7 separable conv
        if action == '7x7 dconv':
            x = ops.SeperableConvolution(filters, (7, 7), strides)
            x._name = f'7x7_dconv_c{self.cell_index}b{self.block_index}{tag}'
            return x

        # applies a 1x7 and then a 7x1 standard conv operation
        if action == '1x7-7x1 conv':
            f = [filters, filters]
            k = [(1, 7), (7, 1)]
            s = [strides, 1]

            x = ops.StackedConvolution(f, k, s)
            x._name = f'1x7-7x1_conv_c{self.cell_index}b{self.block_index}{tag}'
            return x

        # applies a 3x3 standard conv
        if action == '3x3 conv':
            x = ops.Convolution(filters, (3, 3), strides)
            x._name = f'3x3_conv_c{self.cell_index}b{self.block_index}{tag}'
            return x

        # applies a 3x3 maxpool
        if action == '3x3 maxpool':
            x = ops.Pooling('max', (3, 3), strides=strides)
            x._name = f'3x3_maxpool_c{self.cell_index}b{self.block_index}{tag}'
            return x

        # applies a 3x3 avgpool
        if action == '3x3 avgpool':
            x = ops.Pooling('avg', (3, 3), strides=strides)
            x._name = f'3x3_avgpool_c{self.cell_index}b{self.block_index}{tag}'
            return x

        # 'identity' action case, if using (2, 2) stride it's actually handled as a convolution
        # attempts a linear operation (if size matches) or a strided linear conv projection to reduce spatial depth
        if strides == (2, 2):
            x = ops.Convolution(filters, (1, 1), strides)
            x._name = f'pointwise_conv_c{self.cell_index}b{self.block_index}{tag}'
            return x
        else:
            # else just submits a linear layer if shapes match
            x = ops.Identity(filters, strides)
            x._name = f'identity_c{self.cell_index}b{self.block_index}{tag}'
            return x
