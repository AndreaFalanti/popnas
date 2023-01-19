import csv
import operator
import re
from abc import ABC, abstractmethod
from statistics import mean
from typing import Iterable, Callable, NamedTuple, Optional, TypeVar, Any

from typing_extensions import Protocol

import log_service
from search_space import CellSpecification

T = TypeVar("T")


class MaxMinCallable(Protocol):
    def __call__(self, iterable: Iterable[T], key: Callable = None) -> T: ...


class TargetMetric(NamedTuple):
    '''
    Utility class for specifying metrics and how their optimal value among epochs is computed.
    The name must be the one used in the Keras metric, since it is used to extract info from the training history.
    '''
    name: str
    optimal: MaxMinCallable
    results_csv_column: str
    units: Optional[str] = None
    prediction_csv_column: str = ''

    def plot_label(self):
        return self.name + ('' if self.units is None else f' ({self.units})')

    def to_keras_history_key(self, validation: bool, output_name: str = ''):
        key_prefix = 'val_' if validation else ''
        return f'{key_prefix}{output_name}_{self.name}'


def get_pareto_targeted_metrics(all_metrics: 'list[TargetMetric]', pareto_objectives: 'list[str]'):
    return [m for m in all_metrics if m.name in pareto_objectives]


def get_best_metric_in_history(history: 'dict[str, list]', metric_name: str, optimal: MaxMinCallable, multi_output: bool):
    if multi_output:
        multi_output_metric_vals = get_best_metric_per_output(history, metric_name, optimal=optimal)
        return optimal(multi_output_metric_vals.values())
    else:
        return optimal(history[f'val_{metric_name}'])


def extract_metric_from_train_histories(histories: 'list[dict[str, list]]', metric_name: str, optimal: MaxMinCallable,
                                        multi_output: bool):
    return mean(get_best_metric_in_history(hist, metric_name, optimal, multi_output) for hist in histories)


def get_best_metric_and_epoch_index(history: 'dict[str, Any]', metric: TargetMetric,
                                    output_names: 'list[str]', using_val: bool) -> 'tuple[int, float]':
    # get the best metric for each cell, plus the epoch index where they occur
    best_scores = [metric.optimal(enumerate(history[metric.to_keras_history_key(using_val, output_name=output_name)]), key=operator.itemgetter(1))
                   for output_name in output_names]

    return metric.optimal(best_scores, key=operator.itemgetter(1))


def get_best_metric_per_output(history: 'dict[str, list]', metric: str, optimal: MaxMinCallable):
    '''
    Produce a dictionary with a key for each model output.
    The value associated is the best one found for that output during the whole train procedure.
    Args:
        history: train history, the "history" dictionary of the History callback returned by model.fit(...)
        metric: metric name
        optimal: function to return optimal value (max or min)

    Returns:
        (dict[str, float]): dictionary with best metric values, for each output.
    '''
    r = re.compile(rf'val_Softmax_c(\d+)_{metric}')
    output_indexes = [int(match.group(1)) for match in map(r.match, history.keys()) if match]

    # save best score reached for each output
    multi_output_scores = {}
    for output_index in output_indexes:
        multi_output_scores[f'c{output_index}_{metric}'] = optimal(history[f'val_Softmax_c{output_index}_{metric}'])

    return multi_output_scores


def write_multi_output_results_to_csv(file_path: str, cell_spec: CellSpecification, histories: 'list[dict[str, list]]',
                                      metrics: 'list[TargetMetric]', csv_headers: 'list[str]'):
    outputs_dict = {}
    for m in metrics:
        metric_values = []
        for hist in histories:
            metrics_dict = get_best_metric_per_output(hist, m.name, m.optimal)
            metric_values.append(metrics_dict.values())

        # always the same for each metric, so overriding them in each cycle is not a problem
        metric_names = metrics_dict.keys()
        # position-wise mean
        metric_values_mean = [mean(values_of_same_output) for values_of_same_output in zip(*metric_values)]
        outputs_dict = {**outputs_dict, **{k: v for k, v in zip(metric_names, metric_values_mean)}}

    # add cell spec to dictionary and write it into the csv
    outputs_dict['cell_spec'] = str(cell_spec)

    with open(file_path, mode='a+', newline='') as f:
        # append mode, so if file handler is in position 0 it means is empty. In this case, write the headers too.
        if f.tell() == 0:
            writer = csv.writer(f)
            writer.writerow(csv_headers)

        writer = csv.DictWriter(f, csv_headers)
        writer.writerow(outputs_dict)


class BaseTrainingResults(ABC):
    ''' Utility class for passing the training results of a networks, together with other interesting network characteristics. '''

    def __init__(self, cell_spec: CellSpecification, training_time: float, inference_time: float, params: int, flops: int) -> None:
        self.cell_spec = cell_spec
        self.training_time = training_time
        self.inference_time = inference_time
        self.params = params
        self.flops = flops

        self.blocks = len(cell_spec)
        self._logger = log_service.get_logger('TrainingResults')

    @staticmethod
    @abstractmethod
    def from_training_histories(cell_spec: CellSpecification, training_time: float, inference_time: float, params: int, flops: int,
                                histories: 'list[dict[str, list]]', multi_output: bool) -> 'BaseTrainingResults':
        '''
        Instantiate the training results from the histories obtained after multiple training sessions
        (list with a single history are ok for single training sessions).
        '''
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def from_csv_row(row: list) -> 'BaseTrainingResults':
        ''' Instantiate the training results from a csv row. '''
        raise NotImplementedError()

    @staticmethod
    def custom_metrics_considered() -> 'list[TargetMetric]':
        ''' Return all TargetMetric objects related to metrics not included in Keras and computed with custom functions. '''
        return [
            TargetMetric('time', min, results_csv_column='training time(seconds)', units='seconds', prediction_csv_column='time'),
            TargetMetric('inference_time', min, results_csv_column='inference time(seconds)', units='seconds'),
            TargetMetric('params', min, results_csv_column='total params'),
            TargetMetric('flops', min, results_csv_column='flops')
        ]

    @staticmethod
    def keras_metrics_considered() -> 'list[TargetMetric]':
        ''' Return all TargetMetric objects related to Keras metrics, extractable from training history. '''
        return []

    @classmethod
    def all_metrics_considered(cls) -> 'list[TargetMetric]':
        ''' Return all TargetMetric objects considered by the class implementation. '''
        return cls.keras_metrics_considered() + cls.custom_metrics_considered()

    @classmethod
    def predictable_metrics_considered(cls) -> 'list[TargetMetric]':
        ''' Return the TargetMetric objects which can be targeted by predictors
         (metrics not directly computable and potential Pareto optimization targets). '''
        return [m for m in cls.all_metrics_considered() if m.prediction_csv_column]

    @classmethod
    def get_csv_headers(cls) -> 'list[str]':
        return [m.results_csv_column for m in cls.all_metrics_considered()] + ['# blocks', 'cell structure']

    @abstractmethod
    def to_csv_list(self) -> list:
        ''' Return a list with fields ordered for csv insertion. '''
        return [self.training_time, self.inference_time, self.params, self.flops, self.blocks, str(self.cell_spec)]

    @abstractmethod
    def log_results(self):
        ''' Print the results, in a human-readable format, to the logger. '''
        self._logger.info("Training time: %0.4f", self.training_time)
        self._logger.info('Inference time: %0.4f', self.inference_time)
        # format is a workaround for thousands separator, since the python logger has no such feature
        self._logger.info("Total parameters: %s", format(self.params, ','))
        self._logger.info("Total FLOPS: %s", format(self.flops, ','))
