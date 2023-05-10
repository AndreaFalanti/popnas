import os

import tensorflow as tf

import log_service
from models.generators import BaseModelGenerator
from models.results.base import TargetMetric
from predictors.models import *
from search_space import SearchSpace
from search_space_units import CellSpecification
from utils.feature_utils import generate_acc_features, generate_time_features


class PredictorsHandler:
    # TODO: could be extended with predictors configs for hyperparameters
    # TODO: score domain could be put directly in TargetMetric, to increase consistency and flexibility
    def __init__(self, search_space: SearchSpace,
                 score_metric: TargetMetric, score_domain: 'tuple[float, float]', pareto_metrics: 'list[TargetMetric]',
                 model_gen: BaseModelGenerator, train_strategy: tf.distribute.Strategy, pnas_mode: bool = False) -> None:
        '''
        Utility class to initialize the predictors and expose "shortcut" functions for quickly returning the features used by a predictor from any
        cell specification.

        Args:
            search_space: a SearchSpace instance.
            score_metric: the TargetMetric related to the score metric, specified in the POPNAS JSON configuration file.
            score_domain: the numerical interval of the score metric, as a tuple containing the lower and upper bound,
             e.g., for accuracy it's equal to (0, 1).
            model_gen: the model generator used for the task addressed.
            train_strategy: train strategy (device distribution) used for the predictors.
            pnas_mode: True if using PNAS mode (see JSON config).
        '''
        self._search_space = search_space
        self._model_gen = model_gen
        self.pareto_metrics = pareto_metrics
        pareto_metric_names = [m.name for m in self.pareto_metrics]

        score_csv_field = score_metric.results_csv_column
        predictors_log_path = log_service.build_path('predictors')
        # catboost_time_desc_path = log_service.build_path('csv', 'column_desc_time.csv')
        amllibrary_time_config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'amllibrary.ini')
        if not os.path.exists(amllibrary_time_config_path):
            raise FileNotFoundError('aMLLibrary time configuration file not found')
        amllibrary_inference_time_config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'amllibrary_inference.ini')
        if not os.path.exists(amllibrary_inference_time_config_path):
            raise FileNotFoundError('aMLLibrary inference time configuration file not found')

        self._logger = log_service.get_logger(__name__)
        self._logger.info('Initializing predictors...')

        # initialize the score predictors to be used
        self._acc_predictor = GINPredictor(search_space, score_csv_field, score_domain, train_strategy,
                                           self._logger, predictors_log_path, override_logs=False, save_weights=True)

        # initialize the time predictors to be used
        # to use CatBoost: CatBoostPredictor(catboost_time_desc_path, self._logger, predictors_log_path, use_random_search=True, override_logs=False)
        # example for aMLLibrary: AMLLibraryPredictor(amllibrary_config_path, ['LRRidge'], self._logger, predictors_log_path, override_logs=False)
        self._time_predictor = None if pnas_mode or 'time' not in pareto_metric_names else \
            AMLLibraryPredictor(amllibrary_time_config_path, ['SVR'], self._logger, predictors_log_path,
                                override_logs=False, name='aMLLibrary_SVR_training_time')

        self._inference_time_predictor = None if pnas_mode or 'inference_time' not in pareto_metric_names else \
            AMLLibraryPredictor(amllibrary_inference_time_config_path, ['SVR'], self._logger, predictors_log_path,
                                override_logs=False, name='aMLLibrary_SVR_inference_time')

        self._logger.info('Predictors generated successfully')

    # TODO: b (number of blocks) allows to use different predictor based on the step. Actually not used, but preserves the public interface.
    def get_score_predictor(self, b: int):
        return self._acc_predictor

    def get_time_predictor(self, b: int):
        return self._time_predictor

    def get_inference_time_predictor(self, b: int):
        return self._inference_time_predictor

    def generate_cell_score_features(self, cell_spec: CellSpecification):
        return generate_acc_features(cell_spec, self._search_space, self._model_gen.get_real_cell_depth)

    def generate_cell_time_features(self, cell_spec: CellSpecification):
        return generate_time_features(cell_spec, self._search_space, 'dynamic_reindex', self._model_gen.get_real_cell_depth)

    def generate_cell_inference_time_features(self, cell_spec: CellSpecification):
        return generate_time_features(cell_spec, self._search_space, 'dynamic_reindex_inference', self._model_gen.get_real_cell_depth)

    # since it has the reference to the model generator, it's nice to expose the graph generation for agents
    # that need to consider extra constraints, like the params.
    def get_architecture_dag(self, cell_spec: CellSpecification):
        return self._model_gen.build_model_graph(cell_spec)
