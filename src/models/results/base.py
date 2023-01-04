import csv
import re
from abc import ABC, abstractmethod
from statistics import mean
from typing import Iterable, Callable, NamedTuple

from utils.func_utils import cell_spec_to_str


class MetricTarget(NamedTuple):
    '''
    Utility class for specifying metrics and how their optimal value among epochs is computed.
    The name must be the one used in the Keras metric, since it is used to extract info from the training history.
    '''
    name: str
    optimal: Callable[[Iterable], float]


def get_best_metric_in_history(history: 'dict[str, list]', metric_name: str, optimal: Callable[[Iterable], float], multi_output: bool):
    if multi_output:
        multi_output_metric_vals = get_best_metric_per_output(history, metric_name, optimal=optimal)
        return optimal(multi_output_metric_vals.values())
    else:
        return optimal(history[f'val_{metric_name}'])


def extract_metric_from_train_histories(histories: 'list[dict[str, list]]', metric_name: str, optimal: Callable[[Iterable], float],
                                        multi_output: bool):
    return mean(get_best_metric_in_history(hist, metric_name, optimal, multi_output) for hist in histories)


def get_best_metric_per_output(history: 'dict[str, list]', metric: str, optimal: Callable[[Iterable], float]):
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


def write_multi_output_results_to_csv(file_path: str, cell_spec: list, histories: 'list[dict[str, list]]',
                                      metrics: 'list[MetricTarget]', csv_headers: 'list[str]'):
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
    outputs_dict['cell_spec'] = cell_spec_to_str(cell_spec)

    with open(file_path, mode='a+', newline='') as f:
        # append mode, so if file handler is in position 0 it means is empty. In this case, write the headers too.
        if f.tell() == 0:
            writer = csv.writer(f)
            writer.writerow(csv_headers)

        writer = csv.DictWriter(f, csv_headers)
        writer.writerow(outputs_dict)


class BaseTrainingResults(ABC):
    ''' Utility class for passing the training results of a networks, together with other interesting network characteristics. '''

    def __init__(self, cell_spec: 'list[tuple]', training_time: float, inference_time: float, params: int, flops: int) -> None:
        self.cell_spec = cell_spec
        self.training_time = training_time
        self.inference_time = inference_time
        self.params = params
        self.flops = flops

        self.blocks = len(cell_spec)

    @staticmethod
    @abstractmethod
    def from_training_histories(cell_spec: 'list[tuple]', training_time: float, inference_time: float, params: int, flops: int,
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
    @abstractmethod
    def metrics_considered() -> 'list[MetricTarget]':
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def get_csv_headers() -> 'list[str]':
        return ['training time(seconds)', 'inference time(seconds)', 'total params', 'flops', '# blocks', 'cell structure']

    @abstractmethod
    def to_csv_list(self) -> list:
        ''' Return a list with fields ordered for csv insertion. '''
        return [self.training_time, self.inference_time, self.params, self.flops, self.blocks, self.cell_spec]
