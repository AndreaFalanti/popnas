from logging import Logger

from ray import tune
from tensorflow.keras import layers, regularizers, Model

from encoder import SearchSpace
from keras_predictor import KerasPredictor
from predictors.common.datasets_gen import build_temporal_series_dataset_2i


class Conv1DPredictor(KerasPredictor):
    def __init__(self, search_space: SearchSpace, y_col: str, y_domain: 'tuple[float, float]',
                 logger: Logger, log_folder: str, name: str = None, override_logs: bool = True,
                 use_previous_data: bool = True, save_weights: bool = False, hp_config: dict = None, hp_auto_tuning: bool = True):
        # generate a relevant name if not set
        if name is None:
            name = f'Conv1D_{"default" if hp_config is None else hp_config}_{"tune" if hp_auto_tuning else "manual"}'

        super().__init__(y_col, y_domain, logger, log_folder, name, override_logs, use_previous_data, save_weights, hp_config, hp_auto_tuning)

        self.search_space = search_space

    def _get_default_hp_config(self):
        return {
            'epochs': 20,
            'lr': 0.01,
            'wr': 1e-5,
            'filters': 12,
            'kernel_size': 2,
            'dense_units': 10
        }

    def _get_default_hp_search_space(self):
        return {
            'epochs': 20,
            'lr': tune.loguniform(0.01, 0.15),
            'wr': tune.uniform(1e-6, 1e-4),
            'filters': tune.uniform(10, 40),
            'kernel_size': tune.randint(2, 4),
            'dense_units': tune.uniform(5, 40)
        }

    def _build_model(self, config: dict):
        weight_reg = regularizers.l2(config['wr']) if config['wr'] > 0 else None

        # two inputs: one tensor for cell inputs, one for cell operators
        inputs = layers.Input(shape=(self.search_space.B, 2))
        ops = layers.Input(shape=(self.search_space.B, 2))

        inputs_temp_conv = layers.Conv1D(config['filters'], config['kernel_size'], activation='relu', kernel_regularizer=weight_reg)(inputs)
        ops_temp_conv = layers.Conv1D(config['filters'], config['kernel_size'], activation='relu', kernel_regularizer=weight_reg)(ops)

        # indicating [batch_size, serie_length, features(whole block embedding)]
        block_serie = layers.Concatenate()([inputs_temp_conv, ops_temp_conv])

        block_temp_conv = layers.Conv1D(config['filters'], * 2, config['kernel_size'], activation='relu', kernel_regularizer=weight_reg)(block_serie)

        flatten = layers.Flatten()(block_temp_conv)
        sig_dense = layers.Dense(config['dense_units'], activation='sigmoid', kernel_regularizer=weight_reg)(flatten)
        score = layers.Dense(1, activation=self.output_activation, kernel_regularizer=weight_reg)(sig_dense)

        return Model(inputs=(inputs, ops), outputs=score)

    def _build_tf_dataset(self, cell_specs: 'list[list]', rewards: 'list[float]' = None,
                          use_data_augmentation: bool = True, validation_split: bool = True):
        return build_temporal_series_dataset_2i(self.search_space, cell_specs, rewards, validation_split, use_data_augmentation)
