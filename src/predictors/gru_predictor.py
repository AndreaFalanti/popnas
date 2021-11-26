from logging import Logger

import keras_tuner as kt
from tensorflow.keras import layers, regularizers, Model

from encoder import SearchSpace
from keras_predictor import KerasPredictor
from predictors.common.datasets_gen import build_temporal_series_dataset_2i


class GRUPredictor(KerasPredictor):
    def __init__(self, search_space: SearchSpace, y_col: str, y_domain: 'tuple[float, float]',
                 logger: Logger, log_folder: str, name: str = None, override_logs: bool = True,
                 use_previous_data: bool = True, save_weights: bool = False, hp_config: dict = None, hp_auto_tuning: bool = True):
        # generate a relevant name if not set
        if name is None:
            name = f'GRU_{"default" if hp_config is None else hp_config}_{"tune" if hp_auto_tuning else "manual"}'

        super().__init__(y_col, y_domain, logger, log_folder, name, override_logs, use_previous_data, save_weights, hp_config, hp_auto_tuning)

        self.search_space = search_space

    def _get_default_hp_config(self):
        return {
            'epochs': 20,
            'lr': 0.01,
            'wr': 1e-5,
            'cells': 48,
            'embedding_dim': 10
        }

    def _get_hp_search_space(self):
        hp = kt.HyperParameters()
        hp.Fixed('epochs', 20)
        hp.Float('lr', 0.004, 0.04, sampling='linear')
        hp.Float('wr', 1e-7, 1e-4, sampling='log')
        hp.Float('er', 1e-7, 1e-4, sampling='log')
        hp.Int('cells', 20, 100, sampling='linear')
        hp.Int('embedding_dim', 10, 100, sampling='linear')

        return hp

    # TODO: same code of LSTM, if no changes should be done for GRU then set it as a config parameter? Like grid search of LSTM and GRU.
    def _build_model(self, config: dict):
        weight_reg = regularizers.l2(config['wr']) if config['wr'] > 0 else None
        embedding_reg = regularizers.l2(config['er']) if config['er'] > 0 else None

        # two inputs: one tensor for cell inputs, one for cell operators
        inputs = layers.Input(shape=(self.search_space.B, 2))
        ops = layers.Input(shape=(self.search_space.B, 2))

        # input dim is the max integer value present in the embedding + 1.
        inputs_embed = layers.Embedding(input_dim=self.search_space.inputs_embedding_max, output_dim=config['embedding_dim'],
                                        embeddings_regularizer=embedding_reg, mask_zero=True)(inputs)
        ops_embed = layers.Embedding(input_dim=self.search_space.operator_embedding_max, output_dim=config['embedding_dim'],
                                     embeddings_regularizer=embedding_reg, mask_zero=True)(ops)

        embed = layers.Concatenate()([inputs_embed, ops_embed])
        # pass from 4D (None, B, 2, 2 * embedding_dim) to 3D (None, B, 4 * embedding_dim),
        # indicating [batch_size, serie_length, features(whole block embedding)]
        embed = layers.Reshape((self.search_space.B, 4 * config['embedding_dim']))(embed)

        # many-to-one, so must have return_sequences = False (it is by default)
        gru = layers.Bidirectional(layers.GRU(config['cells'], kernel_regularizer=weight_reg, recurrent_regularizer=weight_reg))(embed)
        score = layers.Dense(1, activation=self.output_activation, kernel_regularizer=weight_reg)(gru)

        return Model(inputs=(inputs, ops), outputs=score)

    def _build_tf_dataset(self, cell_specs: 'list[list]', rewards: 'list[float]' = None,
                          use_data_augmentation: bool = True, validation_split: bool = True):
        return build_temporal_series_dataset_2i(self.search_space, cell_specs, rewards, validation_split, use_data_augmentation)
