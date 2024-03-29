import re

from tensorflow.keras.layers import Layer

from models.operators.layers import *
from models.operators.params_utils import compute_batch_norm_params
from .base import BaseOpAllocator


class LSTMOpAllocator(BaseOpAllocator):
    def compile_op_regex(self) -> re.Pattern:
        return re.compile(r'lstm')

    def compute_params(self, match: re.Match, input_filters: int, output_filters: int) -> int:
        # lstm params + batch normalization that follows the rnn
        return 4 * (output_filters * input_filters + output_filters ** 2 + output_filters) + compute_batch_norm_params(output_filters)

    def generate_normal_layer(self, match: re.Match, filters: int, name_prefix: str = '', name_suffix: str = '') -> Layer:
        return self.generate_depth_adaptation_layer(match, filters, name_prefix, name_suffix)

    def generate_depth_adaptation_layer(self, match: re.Match, filters: int, name_prefix: str = 'D/', name_suffix: str = '') -> Lstm:
        layer_name = f'{name_prefix}lstm{name_suffix}'
        return Lstm(filters, weight_reg=self.weight_reg, name=layer_name)

    def generate_reduction_layer(self, match: re.Match, filters: int, strides: 'tuple[int, ...]',
                                 name_prefix: str = 'R/', name_suffix: str = '') -> RnnBatchReduce:
        rnn = self.generate_depth_adaptation_layer(match, filters, name_prefix, name_suffix)
        return RnnBatchReduce(rnn, strides)


class GRUOpAllocator(BaseOpAllocator):
    def compile_op_regex(self) -> re.Pattern:
        return re.compile(r'gru')

    def compute_params(self, match: re.Match, input_filters: int, output_filters: int) -> int:
        # gru params + batch normalization that follows the rnn
        return 3 * (output_filters * input_filters + output_filters ** 2 + 2 * output_filters) + compute_batch_norm_params(output_filters)

    def generate_normal_layer(self, match: re.Match, filters: int, name_prefix: str = '', name_suffix: str = '') -> Layer:
        return self.generate_depth_adaptation_layer(match, filters, name_prefix, name_suffix)

    def generate_depth_adaptation_layer(self, match: re.Match, filters: int, name_prefix: str = 'D/', name_suffix: str = '') -> Gru:
        layer_name = f'{name_prefix}gru{name_suffix}'
        return Gru(filters, weight_reg=self.weight_reg, name=layer_name)

    def generate_reduction_layer(self, match: re.Match, filters: int, strides: 'tuple[int, ...]',
                                 name_prefix: str = 'R/', name_suffix: str = '') -> RnnBatchReduce:
        rnn = self.generate_depth_adaptation_layer(match, filters, name_prefix, name_suffix)
        return RnnBatchReduce(rnn, strides)
