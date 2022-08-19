'''
Prototype for the Convolutional Vision Transformer "stage", presented in the paper: https://arxiv.org/abs/2103.15808.
The implementation is inspired from: https://github.com/taki0112/vit-tensorflow/blob/main/vit_tensorflow/cvt.py,
but heavily refactored to simplify the implementation and integration in POPNAS.
TODO: still under inspection, since some implementation logic is unclear. Should be better integrated in model too
 (more parametrization in JSON config or dynamically tune the heads / block based on depth).
'''

import tensorflow as tf
from einops import rearrange
from tensorflow import einsum
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, Dropout, Softmax


class LayerNorm(Layer):  # layernorm, but done in the channel dimension #1
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps

        self.g = tf.Variable(tf.ones([1, 1, 1, dim]))
        self.b = tf.Variable(tf.zeros([1, 1, 1, dim]))

    def call(self, x, training=None, **kwargs):
        var = tf.math.reduce_variance(x, axis=-1, keepdims=True)
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)

        x = (x - mean) / tf.sqrt((var + self.eps)) * self.g + self.b
        return x


class PreNorm(Layer):
    def __init__(self, dim, fn):
        super().__init__()

        self.norm = LayerNorm(dim)
        self.fn = fn

    def call(self, x, training=None, **kwargs):
        return self.fn(self.norm(x), training=training)


class MLP(Layer):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()

        self.net = [
            Conv2D(filters=dim * mult, kernel_size=1, strides=1, activation='gelu'),
            Dropout(rate=dropout),
            Conv2D(filters=dim, kernel_size=1, strides=1),
            Dropout(rate=dropout)
        ]
        self.net = Sequential(self.net)

    def call(self, x, training=None, **kwargs):
        return self.net(x, training=training)


class DepthWiseConv2d(Layer):
    def __init__(self, dim_in, dim_out, kernel_size, stride, bias=True):
        super().__init__()

        self.net = Sequential([
            Conv2D(filters=dim_in, kernel_size=kernel_size, strides=stride, padding='SAME', groups=dim_in, use_bias=bias),
            BatchNormalization(momentum=0.9, epsilon=1e-5),
            Conv2D(filters=dim_out, kernel_size=1, strides=1, use_bias=bias)
        ])

    def call(self, x, training=None, **kwargs):
        x = self.net(x, training=training)
        return x


class MultiHeadSelfAttention(Layer):
    def __init__(self, dim, proj_kernel, kv_proj_stride, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = Softmax()

        self.to_q = DepthWiseConv2d(dim, inner_dim, proj_kernel, stride=1, bias=False)
        self.to_kv = DepthWiseConv2d(dim, inner_dim * 2, proj_kernel, stride=kv_proj_stride, bias=False)

        self.to_out = Sequential([
            Conv2D(filters=dim, kernel_size=1, strides=1),
            Dropout(rate=dropout)
        ])

    def call(self, x, training=None, **kwargs):
        b, _, y, n = x.shape
        h = self.heads
        q = self.to_q(x, training=training)
        kv = self.to_kv(x, training=training)
        k, v = tf.split(kv, num_or_size_splits=2, axis=-1)
        qkv = (q, k, v)
        q, k, v = map(lambda t: rearrange(t, 'b x y (h d) -> (b h) (x y) d', h=h), qkv)

        dots = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = self.attend(dots)

        x = einsum('b i j, b j d -> b i d', attn, v)
        x = rearrange(x, '(b h) (x y) d -> b x y (h d)', h=h, y=y)
        x = self.to_out(x, training=training)

        return x


class ConvolutionalTransformerBlock(Layer):
    def __init__(self, filters, proj_kernel, kv_proj_stride, num_blocks, heads, dim_head=64, mlp_mult=4, dropout=0.):
        super().__init__()

        self.layers = []
        for _ in range(num_blocks):
            self.layers.append([
                PreNorm(filters, MultiHeadSelfAttention(filters, proj_kernel=proj_kernel, kv_proj_stride=kv_proj_stride, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(filters, MLP(filters, mlp_mult, dropout=dropout))
            ])

    def call(self, x, training=None, **kwargs):
        for i, (attn, ff) in enumerate(self.layers):
            x = attn(x, training=training) + x
            x = ff(x, training=training) + x

        return x


class CVTStage(Layer):
    def __init__(self,
                 emb_dim: int = 64,
                 emb_kernel: int = 7,
                 emb_stride: int = 4,
                 proj_kernel: int = 3,
                 kv_proj_stride: int = 2,
                 heads: int = 1,
                 ct_blocks: int = 1,
                 mlp_mult: int = 4,
                 dropout: float = 0.0,
                 name: str = 'cvt'):
        super().__init__(name=name)

        self.conv_token_embedding = Conv2D(filters=emb_dim, kernel_size=emb_kernel, padding='same', strides=emb_stride)
        self.layer_norm = LayerNorm(emb_dim)
        self.conv_transformer_block = ConvolutionalTransformerBlock(filters=emb_dim, proj_kernel=proj_kernel,
                                                                    kv_proj_stride=kv_proj_stride, num_blocks=ct_blocks, heads=heads,
                                                                    mlp_mult=mlp_mult, dropout=dropout)

    def call(self, inputs, training=None, **kwargs):
        x = self.conv_token_embedding(inputs, training=training)
        x = self.layer_norm(x, training=training)
        return self.conv_transformer_block(x, training=training)
