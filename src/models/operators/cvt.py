# TODO: still under inspection, since some implementation logic is unclear. Should be better integrated in model too
#  (more parametrization in JSON config or dynamically tune the heads / block based on depth).
from typing import Optional

import tensorflow as tf
from einops import rearrange
from tensorflow import einsum
from tensorflow.keras.activations import gelu
from tensorflow.keras.layers import Layer, Conv2D, DepthwiseConv2D, BatchNormalization, Dropout, Softmax, LayerNormalization
from tensorflow.keras.regularizers import Regularizer


class NullableDropout(Layer):
    ''' A custom dropout layer which become an Identity if rate is set to 0.0. It should improve time efficiency. '''
    def __init__(self, rate: float):
        super().__init__()
        self.rate = rate

        self.drop = Dropout(rate=rate) if rate > 0.0 else tf.identity

    def call(self, x, training=None, **kwargs):
        return self.drop(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            'rate': self.rate
        })
        return config


class PreNorm(Layer):
    def __init__(self, fn):
        super().__init__()

        self.norm = LayerNormalization(epsilon=1e-5)
        self.fn = fn

    def call(self, x, training=None, **kwargs):
        return self.fn(self.norm(x), training=training)


class MLP(Layer):
    def __init__(self, dim, mult=4, dropout=0.0, weight_reg=None):
        super().__init__()
        self.dim = dim
        self.mult = mult
        self.dropout = dropout
        self.weight_reg = weight_reg

        # TODO: the real CvT implementation seems to use dense instead of pointwise conv in MLP.
        #  But a reshape should follow, investigate.
        self.conv_1 = Conv2D(filters=dim * mult, kernel_size=1, strides=1, kernel_initializer='he_uniform', kernel_regularizer=weight_reg)
        self.drop_1 = NullableDropout(rate=dropout)
        self.conv_2 = Conv2D(filters=dim, kernel_size=1, strides=1, kernel_initializer='he_uniform', kernel_regularizer=weight_reg)
        self.drop_2 = NullableDropout(rate=dropout)

    def call(self, x, training=None, **kwargs):
        x = self.conv_1(x, training=training)
        x = gelu(x, approximate=True)
        x = self.drop_1(x, training=training)
        x = self.conv_2(x, training=training)
        return self.drop_2(x, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'mult': self.mult,
            'dropout': self.dropout,
            'weight_reg': self.weight_reg
        })
        return config


class CustomSeparableConv2D(Layer):
    '''
    Basically the usual Depthwise Separable convolution implemented also in Keras, but with BatchNormalization in between.
    '''
    def __init__(self, dim_out, kernel_size, stride, bias=False, weight_reg=None):
        super().__init__()
        self.dim_out = dim_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.weight_reg = weight_reg

        self.depthwise_conv = DepthwiseConv2D(kernel_size=kernel_size, strides=stride, padding='same',
                                              depthwise_initializer='he_uniform', kernel_regularizer=weight_reg, use_bias=bias)
        self.b_norm = BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.pointwise_conv = Conv2D(filters=dim_out, kernel_size=1, strides=1, kernel_initializer='he_uniform',
                                     kernel_regularizer=weight_reg, use_bias=bias)

    def call(self, x, training=None, **kwargs):
        x = self.depthwise_conv(x, training=training)
        x = self.b_norm(x, training=training)
        return self.pointwise_conv(x, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            'dim_out': self.dim_out,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'bias': self.bias,
            'weight_reg': self.weight_reg
        })
        return config


class MultiHeadSelfAttention(Layer):
    def __init__(self, dim, proj_kernel, kv_proj_stride, heads=8, dim_head=64, dropout=0.0, weight_reg=None):
        super().__init__()
        self.dim = dim
        self.proj_kernel = proj_kernel
        self.kv_proj_stride = kv_proj_stride
        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout
        self.weight_reg = weight_reg

        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5

        self.softmax_attn_rescaler = Softmax()

        self.to_q = CustomSeparableConv2D(inner_dim, proj_kernel, stride=1, bias=False, weight_reg=weight_reg)
        self.to_kv = CustomSeparableConv2D(inner_dim * 2, proj_kernel, stride=kv_proj_stride, bias=False, weight_reg=weight_reg)

        # TODO: this is a dense layer in official implementation.
        self.out_conv = Conv2D(filters=dim, kernel_size=1, strides=1, kernel_regularizer=weight_reg)
        self.out_drop = NullableDropout(rate=dropout)

    def call(self, x, training=None, **kwargs):
        # y is needed for rearrange operations
        _, _, y, _ = x.shape
        h = self.heads
        q = self.to_q(x, training=training)
        kv = self.to_kv(x, training=training)
        k, v = tf.split(kv, num_or_size_splits=2, axis=-1)
        qkv = (q, k, v)
        # h is the number of heads, make them scalars (flatten)
        q, k, v = map(lambda t: rearrange(t, 'b x y (h d) -> (b h) (x y) d', h=h), qkv)

        attention_scores = einsum('b i d, b j d -> b i j', q, k) * self.scale
        rescaled_attention = self.softmax_attn_rescaler(attention_scores)

        x = einsum('b i j, b j d -> b i d', rescaled_attention, v)
        x = rearrange(x, '(b h) (x y) d -> b x y (h d)', h=h, y=y)

        x = self.out_conv(x, training=training)
        return self.out_drop(x, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'proj_kernel': self.proj_kernel,
            'kv_proj_stride': self.kv_proj_stride,
            'heads': self.heads,
            'dim_head': self.dim_head,
            'dropout': self.dropout,
            'weight_reg': self.weight_reg
        })
        return config


class ConvolutionalTransformerBlockStack(Layer):
    def __init__(self, filters, proj_kernel, kv_proj_stride, num_blocks, heads, dim_head=64, mlp_mult=4, dropout=0.0, weight_reg=None):
        super().__init__()
        self.filters = filters
        self.proj_kernel = proj_kernel
        self.kv_proj_stride = kv_proj_stride
        self.num_blocks = num_blocks
        self.heads = heads
        self.dim_head = dim_head
        self.mlp_mult = mlp_mult
        self.dropout = dropout
        self.weight_reg = weight_reg

        self.layers = [(
            PreNorm(MultiHeadSelfAttention(filters, proj_kernel, kv_proj_stride, heads, dim_head, dropout=dropout, weight_reg=weight_reg)),
            PreNorm(MLP(filters, mlp_mult, dropout=dropout))
        )] * num_blocks

    def call(self, x, training=None, **kwargs):
        for mhsa, mlp in self.layers:
            x = mhsa(x, training=training) + x
            x = mlp(x, training=training) + x

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'proj_kernel': self.proj_kernel,
            'kv_proj_stride': self.kv_proj_stride,
            'num_blocks': self.num_blocks,
            'heads': self.heads,
            'dim_head': self.dim_head,
            'mlp_mult': self.mlp_mult,
            'dropout': self.dropout,
            'weight_reg': self.weight_reg
        })
        return config


class CVTStage(Layer):
    '''
    Prototype for the Convolutional Vision Transformer "stage", presented in the paper: https://arxiv.org/abs/2103.15808.
    The implementation is inspired from: https://github.com/taki0112/vit-tensorflow/blob/main/vit_tensorflow/cvt.py,
    but heavily refactored to simplify the implementation and integration in POPNAS.
    '''
    def __init__(self, emb_dim: int = 64, emb_kernel: int = 7, emb_stride: int = 4,
                 proj_kernel: int = 3, kv_proj_stride: int = 2, heads: int = 1, dim_head: int = 64,
                 ct_blocks: int = 1, mlp_mult: int = 4, dropout: float = 0.0, weight_reg: Optional[Regularizer] = None, name: str = 'cvt'):
        super().__init__(name=name)
        self.emb_dim = emb_dim
        self.emb_kernel = emb_kernel
        self.emb_stride = emb_stride
        self.proj_kernel = proj_kernel
        self.kv_proj_stride = kv_proj_stride
        self.heads = heads
        self.dim_head = dim_head
        self.ct_blocks = ct_blocks
        self.mlp_mult = mlp_mult
        self.dropout = dropout
        self.weight_reg = weight_reg

        self.conv_token_embedding = Conv2D(filters=emb_dim, kernel_size=emb_kernel, padding='same', strides=emb_stride,
                                           kernel_initializer='he_uniform', kernel_regularizer=weight_reg)
        self.layer_norm = LayerNormalization(epsilon=1e-5)
        self.conv_transformer_block = ConvolutionalTransformerBlockStack(filters=emb_dim, proj_kernel=proj_kernel,
                                                                         kv_proj_stride=kv_proj_stride, num_blocks=ct_blocks,
                                                                         heads=heads, dim_head=dim_head, mlp_mult=mlp_mult,
                                                                         dropout=dropout, weight_reg=weight_reg)

    def call(self, inputs, training=None, **kwargs):
        x = self.conv_token_embedding(inputs, training=training)
        x = self.layer_norm(x, training=training)
        return self.conv_transformer_block(x, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            'emb_dim': self.emb_dim,
            'emb_kernel': self.emb_kernel,
            'emb_stride': self.emb_stride,
            'proj_kernel': self.proj_kernel,
            'kv_proj_stride': self.kv_proj_stride,
            'heads': self.heads,
            'dim_head': self.dim_head,
            'ct_blocks': self.ct_blocks,
            'mlp_mult': self.mlp_mult,
            'dropout': self.dropout,
            'weight_reg': self.weight_reg
        })
        return config


class SimplifiedCVT(Layer):
    '''
    Simplification of CvT, stripped of MLP, basically using (conv_embedding, norm, mhsa), with single head.
    This unit can be used in POPNAS networks without incurring in extremely large time overheads.
    '''
    def __init__(self, emb_dim: int = 64, emb_kernel: int = 7, emb_stride: int = 4,
                 proj_kernel: int = 3, kv_proj_stride: int = 2, heads: int = 1, dim_head: int = 64,
                 dropout: float = 0.0, weight_reg: Optional[Regularizer] = None, name: str = 'simple_cvt'):
        super().__init__(name=name)
        self.emb_dim = emb_dim
        self.emb_kernel = emb_kernel
        self.emb_stride = emb_stride
        self.proj_kernel = proj_kernel
        self.kv_proj_stride = kv_proj_stride
        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout
        self.weight_reg = weight_reg

        self.conv_token_embedding = Conv2D(filters=emb_dim, kernel_size=emb_kernel, padding='same', strides=emb_stride,
                                           kernel_initializer='he_uniform', kernel_regularizer=weight_reg)
        self.layer_norm = LayerNormalization(epsilon=1e-5)
        self.mhsa = MultiHeadSelfAttention(dim=emb_dim, proj_kernel=proj_kernel,
                                           kv_proj_stride=kv_proj_stride, heads=heads,
                                           dim_head=dim_head, dropout=dropout, weight_reg=weight_reg)

    def call(self, inputs, training=None, **kwargs):
        x = self.conv_token_embedding(inputs, training=training)
        x = self.layer_norm(x, training=training)
        return self.mhsa(x, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            'emb_dim': self.emb_dim,
            'emb_kernel': self.emb_kernel,
            'emb_stride': self.emb_stride,
            'proj_kernel': self.proj_kernel,
            'kv_proj_stride': self.kv_proj_stride,
            'heads': self.heads,
            'dim_head': self.dim_head,
            'dropout': self.dropout,
            'weight_reg': self.weight_reg
        })
        return config