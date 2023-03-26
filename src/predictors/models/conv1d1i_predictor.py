from typing import Sequence, Optional

import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model

from predictors.common.datasets_gen import build_temporal_series_dataset
from .keras_predictor import KerasPredictor


class Conv1D1IPredictor(KerasPredictor):
    def _get_default_hp_config(self):
        return dict(super()._get_default_hp_config(), **{
            'filters': 12,
            'kernel_size': 2,
            'dense_units': 10
        })

    def _get_hp_search_space(self):
        hp = super()._get_hp_search_space()
        hp.Int('filters', 10, 40, step=2, sampling='uniform')
        hp.Choice('kernel_size', [2, 3])
        hp.Int('dense_units', 5, 40, sampling='linear')

        return hp

    def _build_model(self, config: dict):
        weight_reg = regularizers.l2(config['wr']) if config['use_wr'] else None

        # one input: a series of blocks
        block_series = layers.Input(shape=(self.search_space.B, 4), dtype=self.dtype)

        first_conv = layers.Conv1D(config['filters'], config['kernel_size'], activation='relu',
                                   kernel_regularizer=weight_reg, padding='same', dtype=self.dtype)(block_series)
        second_conv = layers.Conv1D(config['filters'], config['kernel_size'], activation='relu',
                                    kernel_regularizer=weight_reg, padding='same', dtype=self.dtype)(first_conv)

        final_conv = layers.Conv1D(config['filters'] * 2, config['kernel_size'], activation='relu',
                                   kernel_regularizer=weight_reg, strides=2, dtype=self.dtype)(second_conv)

        flatten = layers.Flatten()(final_conv)
        sig_dense = layers.Dense(config['dense_units'], activation='sigmoid', kernel_regularizer=weight_reg, dtype=self.dtype)(flatten)
        score = layers.Dense(1, activation=None, kernel_regularizer=weight_reg, dtype=self.dtype)(sig_dense)
        out = layers.Activation(self.output_activation, dtype=self.dtype)(score)

        return Model(inputs=block_series, outputs=out)

    def _build_tf_dataset(self, cell_specs: 'Sequence[list]', rewards: 'Sequence[float]' = None, batch_size: int = 8,
                          use_data_augmentation: bool = True, validation_split: bool = True,
                          shuffle: bool = True) -> 'tuple[tf.data.Dataset, Optional[tf.data.Dataset]]':
        return build_temporal_series_dataset(self.search_space, cell_specs, rewards, batch_size, validation_split, use_data_augmentation, shuffle)
