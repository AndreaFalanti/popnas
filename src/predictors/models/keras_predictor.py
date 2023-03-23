import os
from abc import abstractmethod, ABC
from logging import Logger
from statistics import mean
from typing import Union, Any, Optional, Sequence

import keras
import keras_tuner as kt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow.keras import losses, optimizers, metrics, callbacks
from tensorflow.keras.utils import plot_model
from tensorflow_addons import metrics as tfa_metrics
from tensorflow_addons import optimizers as tfa_optimizers

from search_space import SearchSpace, parse_cell_strings
from utils.func_utils import create_empty_folder, alternative_dict_to_string
from utils.nn_utils import get_optimized_steps_per_execution
from utils.rstr import rstr
from .predictor import Predictor


class KerasPredictor(Predictor, ABC):
    def __init__(self, search_space: SearchSpace, y_col: str, y_domain: 'tuple[float, float]', train_strategy: tf.distribute.Strategy, logger: Logger, log_folder: str,
                 name: str = None, override_logs: bool = True, save_weights: bool = False, hp_config: dict = None, hp_tuning: bool = False):
        # generate a relevant name if not set
        if name is None:
            config_subfix = "default" if hp_config is None else alternative_dict_to_string(hp_config)
            name = f'{self.__class__.__name__}_{config_subfix}_{"tune" if hp_tuning else "manual"}'

        super().__init__(logger, log_folder, name, override_logs)

        # often used for generating features from the possible operators and inputs
        self.search_space = search_space

        self.hp_config = self._get_default_hp_config()
        if hp_config is not None:
            self.hp_config.update(hp_config)

        self.y_col = y_col
        self.save_weights = save_weights

        # TODO: multi-GPU is bugged for predictors, probably related only to RNN layers,
        #  see this: https://github.com/tensorflow/tensorflow/issues/45594.
        #  As workaround, put the model only on the first GPU, the training is very fast anyway.
        self.train_strategy = train_strategy if not isinstance(train_strategy, tf.distribute.MirroredStrategy) \
            else tf.distribute.OneDeviceStrategy(device="/gpu:0")

        self.hp_tuning = hp_tuning

        # choose the correct activation for last layer, based on y domain
        self.output_activation = None
        lower_bound, upper_bound = y_domain
        assert lower_bound < upper_bound, 'Invalid domain'

        # from tests relu is bad for (0, inf) domains. If you want to do more tests, check infinite bounds with math.isinf()
        if lower_bound == 0 and upper_bound == 1:
            self.output_activation = 'sigmoid'
        elif lower_bound == -1 and upper_bound == 1:
            self.output_activation = 'tanh'
        else:
            self.output_activation = 'linear'

        self._logger.debug('Using %s as final activation, based on y domain provided', self.output_activation)

        self._model_log_folder = None  # type: Optional[str]

    @abstractmethod
    def _get_default_hp_config(self) -> 'dict[str, Any]':
        return {
            'epochs': 80,
            'lr': 0.005,
            'wd': 5e-4,
            'use_wr': False,
            'wr': 0
        }

    @abstractmethod
    def _get_hp_search_space(self) -> kt.HyperParameters:
        '''
        Returns a dictionary for sampling each hyperparameter search space.
        Note that keras-tuner HyperParameters class can be treated as a dictionary.
        '''
        hp = kt.HyperParameters()
        hp.Fixed('epochs', 60)
        hp.Float('lr', 0.003, 0.03, sampling='linear')
        hp.Float('wd', 1e-6, 1e-2, sampling='log')
        hp.Boolean('use_wr')
        hp.Float('wr', 1e-7, 1e-4, sampling='log', parent_name='use_wr', parent_values=[True])

        return hp

    @abstractmethod
    def _build_model(self, config: dict) -> keras.Model:
        pass

    @abstractmethod
    def _build_tf_dataset(self, cell_specs: 'Sequence[list]', rewards: 'Sequence[float]' = None, batch_size: int = 8,
                          use_data_augmentation: bool = True, validation_split: bool = True,
                          shuffle: bool = True) -> 'tuple[tf.data.Dataset, Optional[tf.data.Dataset]]':
        '''
        Build a dataset to be used in the Keras predictor.

        Args:
            cell_specs: List of lists of inputs and operators, specification of cells in value form (no encoding).
            rewards: List of rewards (y labels). Defaults to None, provide it for building
                a dataset for training purposes.
            batch_size: Dataset batch size
            use_data_augmentation: flag for enabling data augmentation. Data augmentation simply insert in the dataset some equivalent cell
                representation, aimed to make the neural network to generalize better.
            validation_split: True to reserve 10% of samples for validation purposes, False to not use validation.
            shuffle: shuffle the dataset. Set it to False in prediction to maintain order.

        Returns:
            (tuple[tf.data.Dataset, tf.data.Dataset]): training and validation datasets. Validation dataset is None if validation split is False.
        '''
        pass

    def _get_compilation_parameters(self, lr: float, wd: float,
                                    training_steps: int) -> 'tuple[losses.Loss, optimizers.Optimizer, list[metrics.Metric]]':
        loss = losses.MeanSquaredError()
        train_metrics = [metrics.MeanAbsolutePercentageError(), tfa_metrics.SpearmansRank(name='spearmans_rank')]
        lr = optimizers.schedules.CosineDecay(lr, decay_steps=training_steps)
        wd = optimizers.schedules.CosineDecay(wd, decay_steps=training_steps)
        optimizer = tfa_optimizers.AdamW(learning_rate=lr, weight_decay=wd)

        return loss, optimizer, train_metrics

    def _get_callbacks(self, log_folder: str) -> 'list[callbacks.Callback]':
        return [
            callbacks.TensorBoard(log_dir=log_folder, profile_batch=0, histogram_freq=0, update_freq='epoch', write_graph=False),
            callbacks.EarlyStopping(monitor='loss', patience=12, verbose=1, mode='min', restore_best_weights=True)
        ]

    # kt.HyperParameters is basically a dictionary and can be treated as such
    def _compile_model(self, config: Union[kt.HyperParameters, dict], training_steps: int):
        with self.train_strategy.scope():
            loss, optimizer, train_metrics = self._get_compilation_parameters(config['lr'], config['wd'], training_steps)

            model = self._build_model(config)
            execution_steps = get_optimized_steps_per_execution(self.train_strategy)
            model.compile(optimizer=optimizer, loss=loss, metrics=train_metrics, steps_per_execution=execution_steps)

        return model

    def _get_max_b(self, df: pd.DataFrame):
        return df['# blocks'].max()

    def _extrapolate_samples_for_b(self, training_data_df: pd.DataFrame, b: int, keep_previous: bool = False):
        b_df = training_data_df[training_data_df['# blocks'] <= b] if keep_previous\
            else training_data_df[training_data_df['# blocks'] == b]

        cells = parse_cell_strings(b_df['cell structure'])

        # just return two lists: one with the target, one with the cell structures
        return b_df[self.y_col].to_list(), cells

    def _get_training_data_from_file(self, file_path: str):
        results_df = pd.read_csv(file_path)
        cells = parse_cell_strings(results_df['cell structure'])

        return list(zip(cells, results_df[self.y_col].to_list()))

    # TODO: currently not used, need adjustments for ensemble and new folder structure
    def restore_weights(self):
        if os.path.exists(os.path.join(self.log_folder, 'weights.index')):
            self.model.load_weights(os.path.join(self.log_folder, 'weights'))
            self._logger.info('Weights restored successfully')

    def train_ensemble(self, dataset: Union[str, 'list[tuple]'], splits: int = 5, use_data_augmentation=True):
        # get samples for file path string
        if isinstance(dataset, str):
            dataset = self._get_training_data_from_file(dataset)

        cells, rewards = zip(*dataset)
        model_ensemble = []
        max_b = len(cells[-1])

        # TODO
        cells = list(cells)
        rewards = list(rewards)

        self._logger.info('Starting Keras predictor ensemble training...')

        # TODO: could use test in some way. PNAS ensemble simply trains on 4/5 of the dataset
        # train and test are array of indexes
        for fold_index, (train, test) in enumerate(KFold(n_splits=splits, shuffle=True).split(cells, rewards)):
            self._logger.info('Training ensemble model %d...', fold_index + 1)
            fold_cells = [cells[i] for i in train]
            fold_rewards = [rewards[i] for i in train]
            train_dataset = list(zip(fold_cells, fold_rewards))

            self._model_log_folder = os.path.join(self.log_folder, f'B{max_b}', f'model_{fold_index + 1}')
            create_empty_folder(self._model_log_folder)

            self.train(train_dataset, use_data_augmentation)
            model_ensemble.append(self.model)
            self._logger.info('Model %d training complete', fold_index + 1)

        self.model_ensemble = model_ensemble

    def train(self, dataset: Union[str, 'list[tuple]'], use_data_augmentation=True):
        # get samples for file path string
        if isinstance(dataset, str):
            dataset = self._get_training_data_from_file(dataset)

        cells, rewards = zip(*dataset)
        actual_b = max(len(cell) for cell in cells)
        # erase ensemble in case single training is used. If called from train_ensemble, the ensemble is local to the function and written
        # only at the end
        self.model_ensemble = None

        batch_size = 16
        train_ds, val_ds = self._build_tf_dataset(cells, rewards, batch_size,
                                                  use_data_augmentation=use_data_augmentation, validation_split=self.hp_tuning)
        train_steps_per_epoch = len(train_ds)
        total_training_steps = train_steps_per_epoch * self.hp_config['epochs']

        if self._model_log_folder is None:
            self._model_log_folder = os.path.join(self.log_folder, f'B{actual_b}')
            create_empty_folder(self._model_log_folder)

        train_callbacks = self._get_callbacks(self._model_log_folder)
        self._logger.info('Starting Keras predictor training')

        if self.hp_tuning:
            tuner_callbacks = [callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1, mode='min', restore_best_weights=True)]
            tuner = kt.Hyperband(self._compile_model, objective='val_loss', hyperparameters=self._get_hp_search_space(),
                                 max_epochs=30,
                                 directory=os.path.join(self.log_folder, 'keras-tuner'), project_name=f'B{actual_b}')
            tuner.search(x=train_ds,
                         epochs=self.hp_config['epochs'],
                         validation_data=val_ds,
                         callbacks=tuner_callbacks)
            best_hp = tuner.get_best_hyperparameters()[0].values
            self._logger.info('Best hyperparameters found: %s', rstr(best_hp))

            # train the best model with all samples
            whole_ds = train_ds.concatenate(val_ds)

            self.model = self._compile_model(best_hp, total_training_steps)
            self.model.fit(x=whole_ds,
                           epochs=self.hp_config['epochs'],
                           callbacks=train_callbacks)   # type: callbacks.History
        else:
            self.model = self._compile_model(self.hp_config, total_training_steps)
            self.model.fit(x=train_ds,
                           epochs=self.hp_config['epochs'],
                           callbacks=train_callbacks)

        plot_model(self.model, to_file=os.path.join(self.log_folder, 'model.pdf'), show_shapes=True, show_layer_names=True)
        self._logger.info('Keras predictor trained successfully')

        if self.save_weights:
            self.model.save_weights(os.path.join(self._model_log_folder, 'weights'))

        self._model_log_folder = None

    def predict(self, x: list) -> float:
        pred_dataset, _ = self._build_tf_dataset([x], batch_size=1, validation_split=False, shuffle=False)

        # predict using a single model
        if self.model_ensemble is None:
            return self.model.predict(x=pred_dataset)[0, 0]
        # predict using a model ensemble
        else:
            return mean([model.predict(x=pred_dataset)[0, 0] for model in self.model_ensemble])

    def predict_batch(self, x: 'Sequence[list]') -> np.ndarray:
        pred_dataset, _ = self._build_tf_dataset(x, batch_size=len(x), validation_split=False, shuffle=False)  # preserve order

        # predict using a single model
        if self.model_ensemble is None:
            predictions = self.model.predict(x=pred_dataset, steps=1)  # type: np.ndarray
            return predictions.flatten()
        # predict using a model ensemble
        else:
            predictions = np.array(list(zip(*[model.predict(x=pred_dataset, steps=1) for model in self.model_ensemble]))).squeeze()
            predictions = predictions.mean(axis=-1)
            return predictions.flatten()

    def prepare_prediction_test_data(self, file_path: str) -> 'tuple[list[Union[str, list[tuple]]], list[list], list[list[float]]]':
        dataset_df = pd.read_csv(file_path)
        max_b = self._get_max_b(dataset_df)
        drop_columns = [col for col in dataset_df.columns.values.tolist() if col not in [self.y_col, 'cell structure', '# blocks']]
        dataset_df = dataset_df.drop(columns=drop_columns)

        datasets = []
        prediction_samples = []
        real_values = []

        for b in range(1, max_b):
            targets, cells = self._extrapolate_samples_for_b(dataset_df, b, keep_previous=True)
            datasets.append(list(zip(cells, targets)))

            true_values, cells_to_predict = self._extrapolate_samples_for_b(dataset_df, b + 1)

            real_values.append(true_values)
            prediction_samples.append(cells_to_predict)

        return datasets, prediction_samples, real_values
