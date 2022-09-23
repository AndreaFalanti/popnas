import math

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model


class Conv1DAutoencoder(Model):
    def __init__(self, input_dim: 'tuple[int, ...]', f1: int, f2: int):
        super().__init__()
        # weight_reg = regularizers.l2(1e-6)
        weight_reg = None
        filters = input_dim[-1]  # preserve number of features of the time series
        time_steps = input_dim[0]

        dec_layers = [
            layers.Conv1DTranspose(filters, kernel_size=7, strides=f2, padding='same', activation='relu', kernel_regularizer=weight_reg),
            layers.Conv1DTranspose(filters, kernel_size=7, strides=f1, padding='same', activation='relu', kernel_regularizer=weight_reg),
        ]

        if time_steps % f1 != 0:
            dec_layers.append(layers.Cropping1D(cropping=(0, f1 - time_steps % f1)))
        dim2 = math.ceil(time_steps / f1)
        if dim2 % f2 != 0:
            dec_layers.insert(1, layers.Cropping1D(cropping=(0, f2 - dim2 % f2)))

        self.encoder = tf.keras.Sequential([
            layers.Conv1D(filters, kernel_size=7, strides=f1, padding='same', activation='relu', kernel_regularizer=weight_reg),
            layers.Conv1D(filters, kernel_size=7, strides=f2, padding='same', activation='relu', kernel_regularizer=weight_reg),
        ])
        self.decoder = tf.keras.Sequential(dec_layers)

    def call(self, x, **kwargs):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def get_config(self):
        pass
