import os
from abc import abstractmethod
from logging import Logger
from statistics import mean
from typing import Union, Any, Optional

import keras
import keras_tuner as kt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow.keras import losses, optimizers, metrics, callbacks
from tensorflow.keras.utils import plot_model

from predictors import Predictor
from utils.func_utils import parse_cell_structures, create_empty_folder
from utils.rstr import rstr


class KerasPredictor(Predictor):
    def __init__(self, y_col: str, y_domain: 'tuple[float, float]', logger: Logger, log_folder: str, name: str = None, override_logs: bool = True,
                 save_weights: bool = False, hp_config: dict = None, hp_tuning: bool = False):
        super().__init__(logger, log_folder, name, override_logs)

        self.hp_config = self._get_default_hp_config()
        if hp_config is not None:
            self.hp_config.update(hp_config)

        self.y_col = y_col
        self.save_weights = save_weights

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
            'epochs': 30,
            'lr': 0.004
        }

    @abstractmethod
    def _get_hp_search_space(self) -> kt.HyperParameters:
        '''
        Returns a dictionary for sampling each hyperparameter search space.
        Note that keras-tuner HyperParameters class can be treated as a dictionary.
        '''
        hp = kt.HyperParameters()
        hp.Fixed('epochs', 30)
        hp.Float('lr', 0.002, 0.02, sampling='linear')

        return hp

    @abstractmethod
    def _build_model(self, config: dict) -> keras.Model:
        pass

    @abstractmethod
    def _build_tf_dataset(self, cell_specs: 'list[list]', rewards: 'list[float]' = None, batch_size: int = 8, use_data_augmentation: bool = True,
                          validation_split: bool = True, shuffle: bool = True) -> 'tuple[tf.data.Dataset, tf.data.Dataset]':
        '''
        Build a dataset to be used in the RNN controller.

        Args:
            cell_specs: List of lists of inputs and operators, specification of cells in value form (no encoding).
            rewards: List of rewards (y labels). Defaults to None, provide it for building
                a dataset for training purposes.
            batch_size: Dataset batch size
            use_data_augmentation: flag for enabling data augmentation. Data augmentation simply insert in the dataset some equivalent cell
                representation, aimed to make the neural network to generalize better.
            validation_split: set it to False to use all samples for training, without generating a validation set.
            shuffle: shuffle the dataset. Set it to False in prediction to maintain order.

        Returns:
            (tuple[tf.data.Dataset, tf.data.Dataset]): training and validation datasets. Validation dataset is None if validation split is False.
        '''
        pass

    def _get_compilation_parameters(self, lr: float) -> 'tuple[losses.Loss, optimizers.Optimizer, list[metrics.Metric]]':
        loss = losses.MeanSquaredError()
        train_metrics = [metrics.MeanAbsolutePercentageError()]
        optimizer = optimizers.Adam(learning_rate=lr)

        return loss, optimizer, train_metrics

    def _get_callbacks(self, log_folder: str) -> 'list[callbacks.Callback]':
        return [
            callbacks.TensorBoard(log_dir=log_folder, profile_batch=0, histogram_freq=0, update_freq='epoch'),
            callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1, mode='min', restore_best_weights=True)
        ]

    # kt.HyperParameters is basically a dictionary and can be treated as such
    def _compile_model(self, config: Union[kt.HyperParameters, dict]):
        loss, optimizer, train_metrics = self._get_compilation_parameters(config['lr'])

        model = self._build_model(config)
        model.compile(optimizer=optimizer, loss=loss, metrics=train_metrics)
        return model

    def _get_max_b(self, df: pd.DataFrame):
        return df['# blocks'].max()

    def _extrapolate_samples_for_b(self, training_data_df: pd.DataFrame, b: int, keep_previous: bool = False):
        b_df = training_data_df[training_data_df['# blocks'] <= b] if keep_previous\
            else training_data_df[training_data_df['# blocks'] == b]

        cells = parse_cell_structures(b_df['cell structure'])

        # just return two lists: one with the target, one with the cell structures
        return b_df[self.y_col].to_list(), cells

    def _get_training_data_from_file(self, file_path: str):
        results_df = pd.read_csv(file_path)
        cells = parse_cell_structures(results_df['cell structure'])

        return list(zip(cells, results_df['best val accuracy'].to_list()))

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

        cells = list(cells)
        rewards = list(rewards)

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
        self._model_log_folder = None

    def train(self, dataset: Union[str, 'list[tuple]'], use_data_augmentation=True):
        # get samples for file path string
        if isinstance(dataset, str):
            dataset = self._get_training_data_from_file(dataset)

        cells, rewards = zip(*dataset)
        actual_b = len(cells[-1])
        # erase ensemble in case single training is used. If called from train_ensemble, the ensemble is local to the function and written
        # only at the end
        self.model_ensemble = None

        train_ds, val_ds = self._build_tf_dataset(cells, rewards, use_data_augmentation=use_data_augmentation, validation_split=self.hp_tuning)
        if self._model_log_folder is None:
            b_max = len(cells[-1])
            self._model_log_folder = os.path.join(self.log_folder, f'B{b_max}')
            create_empty_folder(self._model_log_folder)

        train_callbacks = self._get_callbacks(self._model_log_folder)

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

            self.model = self._compile_model(best_hp)
            self.model.fit(x=whole_ds,
                           epochs=self.hp_config['epochs'],
                           callbacks=train_callbacks)   # type: callbacks.History
        else:
            self.model = self._compile_model(self.hp_config)
            self.model.fit(x=train_ds,
                           epochs=self.hp_config['epochs'],
                           callbacks=train_callbacks)

        plot_model(self.model, to_file=os.path.join(self.log_folder, 'model.pdf'), show_shapes=True, show_layer_names=True)
        self._logger.info('Keras predictor trained successfully')

        if self.save_weights:
            self.model.save_weights(os.path.join(self._model_log_folder, 'weights'))

    def predict(self, x: list) -> float:
        pred_dataset, _ = self._build_tf_dataset([x], validation_split=False, shuffle=False)

        # predict using a single model
        if self.model_ensemble is None:
            return self.model.predict(x=pred_dataset)[0, 0]
        # predict using a model ensemble
        else:
            return mean([model.predict(x=pred_dataset)[0, 0] for model in self.model_ensemble])

    def predict_batch(self, x: 'list[list]') -> 'list[float]':
        pred_dataset, _ = self._build_tf_dataset(x, batch_size=len(x), validation_split=False, shuffle=False)  # preserve order

        # predict using a single model
        if self.model_ensemble is None:
            predictions = self.model.predict(x=pred_dataset)  # type: np.ndarray
            return predictions.reshape(-1)
        # predict using a model ensemble
        else:
            predictions = np.array(list(zip(*[model.predict(x=pred_dataset) for model in self.model_ensemble]))).squeeze()
            predictions = predictions.mean(axis=-1)
            return predictions.reshape(-1)

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
