from logging import Logger

from tensorflow.keras import layers, regularizers, Model

from encoder import SearchSpace
from keras_predictor import KerasPredictor
from predictors.common.datasets_gen import build_temporal_series_dataset_2i
from utils.func_utils import alternative_dict_to_string


class RNNPredictor(KerasPredictor):
    def __init__(self, search_space: SearchSpace, y_col: str, y_domain: 'tuple[float, float]',
                 logger: Logger, log_folder: str, name: str = None, override_logs: bool = True,
                 save_weights: bool = False, hp_config: dict = None, hp_tuning: bool = False):
        # generate a relevant name if not set
        if name is None:
            name = f'RNN_{"default" if hp_config is None else alternative_dict_to_string(hp_config)}_{"tune" if hp_tuning else "manual"}'

        super().__init__(y_col, y_domain, logger, log_folder, name, override_logs, save_weights, hp_config, hp_tuning)

        self.search_space = search_space
        self.supported_rnn_classes = {
            'lstm': layers.LSTM,
            'gru': layers.GRU
        }

    def _get_default_hp_config(self):
        return dict(super()._get_default_hp_config(), **{
            'wr': 1e-5,
            'use_er': False,
            'er': 0,
            'cells': 48,
            'embedding_dim': 10,
            'rnn_type': 'lstm'
        })

    def _get_hp_search_space(self):
        hp = super()._get_hp_search_space()
        hp.Float('wr', 1e-7, 1e-4, sampling='log')
        hp.Boolean('use_er')
        hp.Float('er', 1e-7, 1e-4, sampling='log', parent_name='use_er', parent_values=[True])
        hp.Int('cells', 20, 100, sampling='linear')
        hp.Int('embedding_dim', 10, 100, sampling='linear')
        hp.Choice('rnn_type', ['lstm', 'gru'])

        return hp

    def _build_model(self, config: dict):
        weight_reg = regularizers.l2(config['wr']) if config['wr'] > 0 else None
        embedding_reg = regularizers.l2(config['er']) if config['use_er'] else None
        rnn = self.supported_rnn_classes[config['rnn_type']]

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
        lstm = layers.Bidirectional(rnn(config['cells'], kernel_regularizer=weight_reg, recurrent_regularizer=weight_reg))(embed)
        score = layers.Dense(1, activation=self.output_activation, kernel_regularizer=weight_reg)(lstm)

        return Model(inputs=(inputs, ops), outputs=score)

    def _build_tf_dataset(self, cell_specs: 'list[list]', rewards: 'list[float]' = None, batch_size: int = 8, use_data_augmentation: bool = True,
                          validation_split: bool = True, shuffle: bool = True):
        return build_temporal_series_dataset_2i(self.search_space, cell_specs, rewards, batch_size, validation_split, use_data_augmentation, shuffle)
