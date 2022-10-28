import re
from typing import Optional

from tensorflow.keras.layers import Layer
from tensorflow.keras.regularizers import Regularizer

from .base import BaseOpAllocator
from models.operators.layers import *


class LSTMOpAllocator(BaseOpAllocator):
    def compile_op_regex(self) -> re.Pattern:
        return re.compile(r'lstm')

    def generate_normal_layer(self, match: re.Match, filters: int, weight_reg: Optional[Regularizer], name_prefix: str = '',
                              name_suffix: str = '') -> Layer:
        return self.generate_depth_adaptation_layer(match, filters, weight_reg, name_prefix, name_suffix)

    def generate_depth_adaptation_layer(self, match: re.Match, filters: int, weight_reg: Optional[Regularizer], name_prefix: str = 'D/',
                                        name_suffix: str = '') -> Lstm:
        layer_name = f'{name_prefix}lstm{name_suffix}'
        return Lstm(filters, weight_reg=weight_reg, name=layer_name)

    def generate_reduction_layer(self, match: re.Match, filters: int, weight_reg: Optional[Regularizer], strides: 'tuple[int, ...]',
                                 name_prefix: str = 'R/', name_suffix: str = '') -> RnnBatchReduce:
        rnn = self.generate_depth_adaptation_layer(match, filters, weight_reg, name_prefix, name_suffix)
        return RnnBatchReduce(rnn, strides)


class GRUOpAllocator(BaseOpAllocator):
    def compile_op_regex(self) -> re.Pattern:
        return re.compile(r'gru')

    def generate_normal_layer(self, match: re.Match, filters: int, weight_reg: Optional[Regularizer], name_prefix: str = '',
                              name_suffix: str = '') -> Layer:
        return self.generate_depth_adaptation_layer(match, filters, weight_reg, name_prefix, name_suffix)

    def generate_depth_adaptation_layer(self, match: re.Match, filters: int, weight_reg: Optional[Regularizer], name_prefix: str = 'D/',
                                        name_suffix: str = '') -> Gru:
        layer_name = f'{name_prefix}gru{name_suffix}'
        return Gru(filters, weight_reg=weight_reg, name=layer_name)

    def generate_reduction_layer(self, match: re.Match, filters: int, weight_reg: Optional[Regularizer], strides: 'tuple[int, ...]',
                                 name_prefix: str = 'R/', name_suffix: str = '') -> RnnBatchReduce:
        rnn = self.generate_depth_adaptation_layer(match, filters, weight_reg, name_prefix, name_suffix)
        return RnnBatchReduce(rnn, strides)
