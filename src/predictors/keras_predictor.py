import os
from abc import abstractmethod
from logging import Logger
from typing import Union, Any

import keras
import pandas as pd
import tensorflow as tf

from predictors import Predictor
from ray import tune
from ray.tune.integration.keras import TuneReportCallback
from tensorflow.keras import losses, optimizers, metrics, callbacks
from tensorflow.keras.utils import plot_model
from utils.func_utils import parse_cell_structures
from utils.rstr import rstr


class KerasPredictor(Predictor):
    def __init__(self, y_col: str, y_domain: 'tuple[float, float]', logger: Logger, log_folder: str, name: str = None, override_logs: bool = True,
                 use_previous_data: bool = True, save_weights: bool = False, hp_config: dict = None, hp_auto_tuning: bool = True):
        super().__init__(logger, log_folder, name, override_logs)

        self.y_col = y_col
        default_config = self._get_default_hp_search_space() if hp_auto_tuning else self._get_default_hp_config()
        self.hp_config = default_config if hp_config is None else default_config.update(hp_config)
        self.use_previous_data = use_previous_data
        self.save_weights = save_weights

        self.hp_auto_tuning = hp_auto_tuning

        # used to accumulate samples in a common dataset (a list for each B), if use_previous_data is True
        self.children_history = []
        self.score_history = []

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

    @abstractmethod
    def _get_default_hp_config(self) -> 'dict[str, Any]':
        return {
            'epochs': 20,
            'lr': 0.01
        }

    @abstractmethod
    def _get_default_hp_search_space(self) -> 'dict[str, Any]':
        return {
            'epochs': 20,
            'lr': tune.uniform(0.01, 0.15)
        }

    @abstractmethod
    def _build_model(self, config: dict) -> keras.Model:
        pass

    @abstractmethod
    def _build_tf_dataset(self, cell_specs: 'list[list]', rewards: 'list[float]' = None,
                          use_data_augmentation: bool = True, validation_split: bool = True) -> 'tuple[tf.data.Dataset, tf.data.Dataset]':
        '''
        Build a dataset to be used in the RNN controller.

        Args:
            cell_specs: List of lists of inputs and operators, specification of cells in value form (no encoding).
            rewards: List of rewards (y labels). Defaults to None, provide it for building
                a dataset for training purposes.
            use_data_augmentation: flag for enabling data augmentation. Data augmentation simply insert in the dataset some equivalent cell
                representation, aimed to make the neural network to generalize better.
            validation_split: set it to False to use all samples for training, without generating a validation set.

        Returns:
            (tuple[tf.data.Dataset, tf.data.Dataset]): training and validation datasets. Validation dataset is None if validation split is False.
        '''
        pass

    def _get_compilation_parameters(self, lr: float) -> 'tuple[losses.Loss, optimizers.Optimizer, list[metrics.Metric]]':
        loss = losses.MeanSquaredError()
        train_metrics = [metrics.MeanAbsolutePercentageError()]
        optimizer = optimizers.Adam(learning_rate=lr)

        return loss, optimizer, train_metrics

    def _get_callbacks(self) -> 'list[callbacks.Callback]':
        return [
            callbacks.TensorBoard(log_dir=self.log_folder, profile_batch=0, histogram_freq=0, update_freq='epoch'),
            callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1, mode='min', restore_best_weights=True)
        ]

    def _run_training_session(self, cells: list, rewards: list, use_data_augmentation: bool, validation_split: bool):
        # create the dataset using also previous data, if flag is set
        if self.use_previous_data:
            self.children_history.extend(cells)
            self.score_history.extend(rewards)

            train_ds, val_ds = self._build_tf_dataset(self.children_history, self.score_history, use_data_augmentation, validation_split)
        # use only current data
        else:
            train_ds, val_ds = self._build_tf_dataset(cells, rewards, use_data_augmentation, validation_split)

        def run_experiment(config: dict):
            loss, optimizer, train_metrics = self._get_compilation_parameters(config['lr'])
            train_callbacks = self._get_callbacks()
            # if a validation set is present, it means Tune is performing the hp search. Add the callback.
            if validation_split:
                train_callbacks.append(TuneReportCallback())

            self.model = self._build_model(config)
            self.model.compile(optimizer=optimizer, loss=loss, metrics=train_metrics)

            hist = self.model.fit(x=train_ds,
                                  epochs=self.hp_config['epochs'],
                                  validation_data=val_ds,
                                  callbacks=train_callbacks)
            self._logger.info("losses: %s", rstr(hist.history['loss']))

        return run_experiment

    def _get_max_b(self, df: pd.DataFrame):
        return df['# blocks'].max()

    def _extrapolate_samples_for_b(self, training_data_df: pd.DataFrame, b: int):
        b_df = training_data_df[training_data_df['# blocks'] == b]

        cells = parse_cell_structures(b_df['cell structure'])

        # just return two lists: one with the target, one with the cell structures
        return b_df[self.y_col].tolist(), cells

    def restore_weights(self):
        if os.path.exists(os.path.join(self.log_folder, 'weights.index')):
            self.model.load_weights(os.path.join(self.log_folder, 'weights'))
            self._logger.info('Weights restored successfully')

    def train(self, dataset: Union[str, 'list[tuple]'], use_data_augmentation=True):
        # TODO
        if not isinstance(dataset, list):
            raise TypeError('NN supports only samples, conversion from file is a TODO...')

        cells, rewards = zip(*dataset)

        if self.hp_auto_tuning:
            tune_schedule = tune.schedulers.AsyncHyperBandScheduler()
            # resources_dict = {
            #     'cpu': 0,
            #     'gpu': 1
            # }
            analysis = tune.run(self._run_training_session(cells, rewards, use_data_augmentation, validation_split=True),
                                metric='val_loss', mode='min', num_samples=10, scheduler=tune_schedule, config=self.hp_config,
                                local_dir=os.path.join(self.log_folder, 'ray'))
            self._logger.info('Best hyperparameters found: %s', rstr(analysis.best_config))

            # TODO: train with all samples the best model
            self._run_training_session(cells, rewards, use_data_augmentation, validation_split=False)(analysis.best_config)
        else:
            self._run_training_session(cells, rewards, use_data_augmentation, validation_split=False)(self.hp_config)

        plot_model(self.model, to_file=os.path.join(self.log_folder, 'model.png'), show_shapes=True, show_layer_names=True)

        if self.save_weights:
            self.model.save_weights(os.path.join(self.log_folder, 'weights'))

    def predict(self, sample: list) -> float:
        pred_dataset, _ = self._build_tf_dataset([sample], validation_split=False)
        return self.model.predict(x=pred_dataset)[0, 0]

    def prepare_prediction_test_data(self, file_path: str) -> 'tuple[list[Union[str, list[tuple]]], list[list], list[list[float]]]':
        dataset_df = pd.read_csv(file_path)
        max_b = self._get_max_b(dataset_df)
        drop_columns = [col for col in dataset_df.columns.values.tolist() if col not in [self.y_col, 'cell structure', '# blocks']]
        dataset_df = dataset_df.drop(columns=drop_columns)

        datasets = []
        prediction_samples = []
        real_values = []

        for b in range(1, max_b):
            targets, cells = self._extrapolate_samples_for_b(dataset_df, b)
            datasets.append(list(zip(cells, targets)))

            true_values, cells_to_predict = self._extrapolate_samples_for_b(dataset_df, b + 1)

            real_values.append(true_values)
            prediction_samples.append(cells_to_predict)

        return datasets, prediction_samples, real_values
