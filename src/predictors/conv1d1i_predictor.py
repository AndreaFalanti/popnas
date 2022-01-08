from logging import Logger

from tensorflow.keras import layers, regularizers, Model

from encoder import SearchSpace
from keras_predictor import KerasPredictor
from predictors.common.datasets_gen import build_temporal_series_dataset
from utils.func_utils import alternative_dict_to_string


class Conv1D1IPredictor(KerasPredictor):
    def __init__(self, search_space: SearchSpace, y_col: str, y_domain: 'tuple[float, float]',
                 logger: Logger, log_folder: str, name: str = None, override_logs: bool = True,
                 save_weights: bool = False, hp_config: dict = None, hp_tuning: bool = False):
        # generate a relevant name if not set
        if name is None:
            name = f'Conv1D1I_{"default" if hp_config is None else alternative_dict_to_string(hp_config)}_{"tune" if hp_tuning else "manual"}'

        super().__init__(y_col, y_domain, logger, log_folder, name, override_logs, save_weights, hp_config, hp_tuning)

        self.search_space = search_space

    def _get_default_hp_config(self):
        return dict(super()._get_default_hp_config(), **{
            'wr': 1e-5,
            'filters': 12,
            'kernel_size': 2,
            'dense_units': 10
        })

    def _get_hp_search_space(self):
        hp = super()._get_hp_search_space()
        hp.Float('wr', 1e-7, 1e-4, sampling='log')
        hp.Int('filters', 10, 40, step=2, sampling='uniform')
        hp.Choice('kernel_size', [2, 3])
        hp.Int('dense_units', 5, 40, sampling='linear')

        return hp

    def _build_model(self, config: dict):
        weight_reg = regularizers.l2(config['wr']) if config['wr'] > 0 else None

        # one input: a series of blocks
        block_series = layers.Input(shape=(self.search_space.B, 4))

        first_conv = layers.Conv1D(config['filters'], config['kernel_size'], activation='relu',
                                   kernel_regularizer=weight_reg, padding='same')(block_series)
        second_conv = layers.Conv1D(config['filters'], config['kernel_size'], activation='relu',
                                    kernel_regularizer=weight_reg, padding='same')(first_conv)

        final_conv = layers.Conv1D(config['filters'] * 2, config['kernel_size'], activation='relu',
                                        kernel_regularizer=weight_reg, strides=2)(second_conv)

        flatten = layers.Flatten()(final_conv)
        sig_dense = layers.Dense(config['dense_units'], activation='sigmoid', kernel_regularizer=weight_reg)(flatten)
        score = layers.Dense(1, activation=self.output_activation, kernel_regularizer=weight_reg)(sig_dense)

        return Model(inputs=block_series, outputs=score)

    def _build_tf_dataset(self, cell_specs: 'list[list]', rewards: 'list[float]' = None, batch_size: int = 8, use_data_augmentation: bool = True,
                          validation_split: bool = True, shuffle: bool = True):
        return build_temporal_series_dataset(self.search_space, cell_specs, rewards, batch_size, validation_split, use_data_augmentation, shuffle)
