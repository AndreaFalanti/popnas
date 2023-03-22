import os
from abc import ABC, abstractmethod
from logging import Logger
from typing import Union, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

from utils.func_utils import compute_mape, compute_spearman_rank_correlation_coefficient, create_empty_folder
from utils.plotter_utils import plot_squared_scatter_chart


class Predictor(ABC):
    def __init__(self, logger: Logger, log_folder: str, name: str = None, override_logs: bool = True):
        '''
        Abstract class that provides a common interface to all ML and NN predictors tested in POPNAS work.

        Args:
            logger: logger to use
            log_folder: root folder for predictor logs
            name: optional name to identify the predictor. If not provided, a meaningful one will be produced from arguments provided.
        '''
        self._logger = logger
        self.name = name
        self.log_folder = os.path.join(log_folder, name)

        if override_logs:
            create_empty_folder(self.log_folder)
        else:
            os.makedirs(self.log_folder, exist_ok=True)

        self.model = None
        self.model_ensemble = None  # type: Optional[list]

        # initialize metrics used for prediction tests
        # x_real and y_pred are list of lists. Each list represents all values for a given B value.
        self._x_real = []
        self._y_pred = []
        self._mape = []
        self._spearman = []
        self._r_squared = []

    def _compute_pred_test_metrics(self):
        for x_b, y_b in zip(self._x_real, self._y_pred):
            self._mape.append(compute_mape(x_b, y_b))
            self._spearman.append(compute_spearman_rank_correlation_coefficient(x_b, y_b))
            self._r_squared.append(r2_score(x_b, y_b))

    def save_scatter_plot(self, pred_label: str, save_path: str = None):
        legend_labels = [f'B{i + 2} (MAPE: {mape:.3f}%, R^2: {r2:.3f} ρ: {spearman:.3f})'
                         for i, (mape, r2, spearman) in enumerate(zip(self._mape, self._r_squared, self._spearman))]

        x_label = f'Real {pred_label}'
        y_label = f'Predicted {pred_label}'

        fig = plot_squared_scatter_chart(self._x_real, self._y_pred, x_label, y_label, self.name, legend_labels=legend_labels)

        if save_path is None:
            save_path = os.path.join(self.log_folder, 'results')

        plt.savefig(save_path + '.pdf', bbox_inches='tight')

        plt.title(self.name)
        plt.savefig(save_path + '.png', bbox_inches='tight')
        plt.close(fig)

    def get_model(self):
        return self.model

    @abstractmethod
    def train_ensemble(self, dataset: Union[str, 'list[tuple]'], splits: int = 5, use_data_augmentation=True):
        '''
        Trains multiple models, one for each split.

        Args:
            dataset: can be a file path or a list of tuples (sample, label). Some models could work with only one of these two types.
            splits:
            use_data_augmentation:
        '''
        pass

    @abstractmethod
    def train(self, dataset: Union[str, 'list[tuple]'], use_data_augmentation=True):
        '''
        Trains the model.

        Args:
            dataset: can be a file path or a list of tuples (sample, label). Some models could work with only one of these two types.
            use_data_augmentation:
        '''
        pass

    @abstractmethod
    def predict(self, x: list) -> float:
        '''
        Predict a value for x.
        Args:
            x: a single sample. The sample is expected to be a list of features, input of the prediction.

        Returns:
            (float): prediction
        '''
        pass

    @abstractmethod
    def predict_batch(self, x: 'Sequence[list]') -> np.ndarray:
        '''
        Predict a value for a batch of samples.
        Args:
            x: batch of samples. Each samples is expected to be a list of features, input of the prediction.

        Returns:
            (list[float]): predictions
        '''
        pass

    @abstractmethod
    def prepare_prediction_test_data(self, file_path: str) -> 'tuple[list[Union[str, list[tuple]]], list[list], list[list[float]]]':
        '''
        Prepare the data for the prediction test.
        Args:
            file_path (str):

        Returns:
            (tuple): list of datasets (for training), nested list of samples (to predict), nested list of real values of predictions.
            Nested lists are lists of lists where each list represent values inherent to same B blocks num.
        '''
        pass

    def perform_prediction_test(self, file_path: str, plot_label: str, ensemble_count: Optional[int] = None):
        self._logger.info('Starting prediction testing...')
        datasets, samples_to_predict, self._x_real = self.prepare_prediction_test_data(file_path)
        self._logger.info('Data preprocessed successfully')

        b_max = 1 + len(datasets)
        for b, dataset_b, samples_b in zip(range(1, b_max), datasets, samples_to_predict):
            self._logger.info('Starting test procedure on B=%d', b)
            if ensemble_count is not None and ensemble_count > 1:
                self.train_ensemble(dataset_b, splits=ensemble_count)
            else:
                self.train(dataset_b)

            self._y_pred.append(self.predict_batch(samples_b).tolist())

        self._logger.info('Computing additional metrics...')
        self._compute_pred_test_metrics()
        self._logger.info('Build plot about training results...')
        self.save_scatter_plot(plot_label)
        self._logger.info('Plot saved successfully')
