import tensorflow as tf

from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Model


class LSTMAutoencoder(Model):
    def __init__(self, input_len: int, latent_dim: int):
        super().__init__()
        weight_reg = regularizers.l2(1e-4)

        self.encoder = tf.keras.Sequential([
            layers.LSTM(latent_dim * 2, activation='relu', kernel_regularizer=weight_reg, recurrent_regularizer=weight_reg, return_sequences=True),
            layers.LSTM(latent_dim, activation='relu', kernel_regularizer=weight_reg, recurrent_regularizer=weight_reg, return_sequences=True),
        ])
        self.decoder = tf.keras.Sequential([
            layers.LSTM(latent_dim * 2, activation='relu', kernel_regularizer=weight_reg, recurrent_regularizer=weight_reg, return_sequences=True),
            layers.LSTM(input_len, activation='relu', kernel_regularizer=weight_reg, recurrent_regularizer=weight_reg, return_sequences=True),
        ])

    def call(self, x, **kwargs):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def get_config(self):
        pass
