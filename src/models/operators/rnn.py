from abc import abstractmethod, ABC
from typing import Optional

import tensorflow_addons as tfa
from tensorflow.keras.layers import BatchNormalization, Layer, RNN, GRU, LSTM
from tensorflow.keras.regularizers import Regularizer

from models.operators import Pooling


class RnnBatch(Layer, ABC):
    @abstractmethod
    def __init__(self, filters: int, weight_reg: Optional[Regularizer] = None, name='abstract', **kwargs):
        '''
        Abstract utility class used as baseline for any {RNN - Batch Normalization} layer.
        Op attribute must be set to a Keras layer or TF nn operation in all concrete implementations.
        '''
        super().__init__(name=name, **kwargs)
        self.filters = filters
        self.weight_reg = weight_reg

        self.bn = BatchNormalization()

        # concrete implementations must use a valid Keras layer / TF operation, assigning it to this variable during __init__
        self.op = None

    def call(self, inputs, training=None, mask=None):
        x = self.op(inputs)
        return self.bn(x, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'weight_reg': self.weight_reg
        })
        return config


class RnnBatchReduce(Layer):
    def __init__(self, rnn_layer: RnnBatch, strides: 'tuple[int, ...]', **kwargs):
        '''
        Abstract utility class used as baseline for any {Operation - Batch Normalization - Activation} layer.
        Op attribute must be set to a Keras layer or TF nn operation in all concrete implementations.
        '''
        super().__init__(name=f'reduction_{rnn_layer.name}', **kwargs)
        self.strides = strides
        self.rnn_layer = rnn_layer

        self.pool = Pooling('max', size=strides, strides=strides)

    def call(self, inputs, training=None, mask=None):
        x = self.rnn_layer(inputs, training=training)
        return self.pool(x, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            'rnn_layer': self.rnn_layer,
            'strides': self.strides
        })
        return config


class Lstm(RnnBatch):
    def __init__(self, filters: int, weight_reg: Optional[Regularizer] = None, name='Lstm', **kwargs):
        super().__init__(filters, weight_reg, name=name, **kwargs)

        # TODO: layer norm version is incredibly slow (between 10 and 100 times slower), no CUDNN optimization seems available. Use normal LSTM.
        # cells = tfa.rnn.LayerNormLSTMCell(filters, kernel_regularizer=weight_reg, recurrent_regularizer=weight_reg)
        # self.op = RNN(cells, return_sequences=True)
        self.op = LSTM(filters, return_sequences=True, kernel_regularizer=weight_reg, recurrent_regularizer=weight_reg)


class Gru(RnnBatch):
    def __init__(self, filters: int, weight_reg: Optional[Regularizer] = None, name='Gru', **kwargs):
        super().__init__(filters, weight_reg, name=name, **kwargs)

        self.op = GRU(filters, return_sequences=True, kernel_regularizer=weight_reg, recurrent_regularizer=weight_reg)
