from logging import Logger
from statistics import mean
from typing import Union, Sequence

import numpy as np

from .predictor import Predictor


class EnsemblePredictor(Predictor):
    ''' Predictor which incapsulates multiple predictor models in a single macro predictor. '''
    def __init__(self, predictors: 'list[Predictor]', logger: Logger, log_folder: str, name: str = None, override_logs: bool = True):
        super().__init__(logger, log_folder, name, override_logs)

        self.predictors = predictors

    def train_ensemble(self, dataset: Union[str, 'list[tuple]'], splits: int = 5, use_data_augmentation=True):
        for predictor in self.predictors:
            predictor.train_ensemble(dataset, splits, use_data_augmentation)

    def train(self, dataset: Union[str, 'list[tuple]'], use_data_augmentation=True):
        for predictor in self.predictors:
            predictor.train(dataset, use_data_augmentation)

    def predict(self, x: list) -> float:
        return mean([predictor.predict(x) for predictor in self.predictors])

    def predict_batch(self, x: 'Sequence[list]') -> np.ndarray:
        predictions = np.stack([predictor.predict_batch(x) for predictor in self.predictors], axis=0)
        # "column-wise" mean, so the mean of the predictions made by all predictors for each input x
        return np.mean(predictions, axis=0)

    def prepare_prediction_test_data(self, file_path: str) -> 'tuple[list[Union[str, list[tuple]]], list[list], list[list[float]]]':
        # TODO: hack implementation, this function is actually ambiguous among multiple predictors,
        #  they must use the same feature format otherwise it will fail
        return self.predictors[0].prepare_prediction_test_data(file_path)
