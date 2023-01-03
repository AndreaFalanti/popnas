from tensorflow.keras.layers import Conv2D, Conv2DTranspose, SeparableConv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, UpSampling2D, \
    Conv1D, Conv1DTranspose, SeparableConv1D, MaxPooling1D, AveragePooling1D, GlobalAveragePooling1D, UpSampling1D

# TODO: it could be nice to import only the correct ones and rename them so that are recognized by layers,
#  but input dims are given at runtime, making it difficult (inner classes with delayed import seems not possible)
op_dim_selector = {
    'conv': {1: Conv1D, 2: Conv2D},
    'tconv': {1: Conv1DTranspose, 2: Conv2DTranspose},
    'dconv': {1: SeparableConv1D, 2: SeparableConv2D},
    'max_pool': {1: MaxPooling1D, 2: MaxPooling2D},
    'avg_pool': {1: AveragePooling1D, 2: AveragePooling2D},
    'gap': {1: GlobalAveragePooling1D, 2: GlobalAveragePooling2D},
    'upsample': {1: UpSampling1D, 2: UpSampling2D}
}

from .common import Identity, SeparableConvolution, Convolution, DilatedConvBatchActivationPooling, StackedConvolution, \
    TransposeConvolution, TransposeConvolutionStack, Pooling, PoolingConv
from .cvt import CVTStage, SimplifiedCVT
from .sdp import ScheduledDropPath
from .rnn import Lstm, Gru, RnnBatchReduce
from .squeeze_excitation import SqueezeExcitation


__all__ = ['Identity', 'SeparableConvolution', 'Convolution', 'StackedConvolution', 'TransposeConvolution', 'TransposeConvolutionStack',
           'DilatedConvBatchActivationPooling', 'Pooling', 'PoolingConv', 'ScheduledDropPath', 'CVTStage', 'SimplifiedCVT', 'Lstm', 'Gru',
           'RnnBatchReduce', 'SqueezeExcitation', 'op_dim_selector']
