import csv
import json
import os
import re
from typing import Union, Sequence

import pandas as pd

from models.generators.factory import get_model_generator_class_for_task
from models.results.base import TargetMetric
from search_space import CellSpecification
from utils.config_dataclasses import RunConfig
from utils.config_utils import retrieve_search_config
from utils.func_utils import from_seconds_to_hms


class SearchInfo:
    def __init__(self, dataset_name: str, num_networks: int, top_score: float, mean5_top_score: float, search_time: float, score_metric: str):
        self.dataset_name = dataset_name
        self.num_networks = num_networks
        self.top_score = top_score
        self.mean5_top_score = mean5_top_score
        self.search_time_seconds = search_time
        self.search_time_hours = search_time / 3600
        self.search_time_str = from_seconds_to_time_str(search_time)
        self.score_metric = score_metric

    @staticmethod
    def from_log_folder(log_folder: str):
        ''' Extract the search procedure info from the provided experiment output folder. '''
        run_config = retrieve_search_config(log_folder)
        score_metric = extract_score_metric(run_config)

        search_time = get_total_search_time(log_folder)
        networks_count = get_sampled_networks_count(log_folder)
        ds_name = get_dataset_name(log_folder)
        best_acc, avg_top5_cells_acc = get_top_scores(log_folder, score_metric)

        return SearchInfo(ds_name, networks_count, best_acc, avg_top5_cells_acc, search_time, score_metric.name)

    def csv_headers(self):
        return ['Dataset', '# Networks', f'Top {self.score_metric}', f'Top-5 Cells {self.score_metric}', 'Search Time']

    def to_csv_row(self):
        return [self.dataset_name, self.num_networks, format(self.top_score, '.3f'), format(self.mean5_top_score, '.3f'), self.search_time_str]


class FinalTrainingInfo:
    def __init__(self, dataset_name: str, cell_spec: CellSpecification, params: int, motifs: int, normal_cells_per_motif: int, filters: int,
                 score: float, training_time: float, score_metric: str):
        self.dataset_name = dataset_name
        self.cell_spec = cell_spec
        self.blocks = len(cell_spec)
        self.params = params
        self.motifs = motifs
        self.normal_cells_per_motif = normal_cells_per_motif
        self.filters = filters
        self.score = score
        self.training_time_seconds = training_time
        self.training_time_hours = training_time / 3600
        self.training_time_str = from_seconds_to_time_str(training_time)
        self.score_metric = score_metric

    @staticmethod
    def from_log_folder(log_folder: str):
        ''' Extract the final training info from the provided experiment output folder. '''
        run_config = retrieve_search_config(log_folder)
        score_metric = extract_score_metric(run_config)

        train_time, params = get_final_network_training_time_and_params(log_folder)
        f, m, n = get_final_network_macro_structure(log_folder)
        ds_name = get_dataset_name(log_folder)
        cell_spec = get_final_network_cell_spec(log_folder)
        acc = get_final_network_accuracy(log_folder)

        return FinalTrainingInfo(ds_name, cell_spec, params, m, n, f, acc, train_time, score_metric.name)

    def csv_headers(self):
        return ['Dataset', 'B', 'M', 'N', 'F', 'Params(M)', self.score_metric, 'Training Time', 'Cell specification']

    def to_csv_row(self):
        return [self.dataset_name, self.blocks, self.motifs, self.normal_cells_per_motif, self.filters, format(self.params / 1e6, '.2f'),
                format(self.score, '.3f'), self.training_time_str, str(self.cell_spec)]


def extract_score_metric(config: RunConfig):
    task = config.dataset.type
    score_metric = config.search_strategy.score_metric
    results_processor = get_model_generator_class_for_task(task).get_results_processor_class()
    return next(m for m in results_processor.keras_metrics_considered() if m.name == score_metric)


def from_seconds_to_time_str(total_seconds: float):
    hours, minutes, seconds = from_seconds_to_hms(total_seconds)
    return f'{hours}h {minutes}m' if hours > 0 else f'{minutes}m {seconds}s'


def get_total_search_time(log_path: str):
    with open(os.path.join(log_path, 'debug.log'), 'r') as f:
        lines = f.read().splitlines()
        last_lines = lines[-4:]

    for line in last_lines:
        match = re.search(r'Total run time: (\d+.\d+)', line)
        if match:
            # in seconds
            return float(match.group(1))


def get_sampled_networks_count(log_path: str):
    # the number of lines in csv is equivalent to the number of networks trained during the run (-1 for headers line)
    with open(os.path.join(log_path, 'csv', 'training_results.csv'), 'r') as f:
        trained_cnn_count = len(f.readlines()) - 1

    return trained_cnn_count


def get_dataset_name(log_path: str):
    run_config = retrieve_search_config(log_path)
    return run_config.dataset.name


def get_top_scores(log_path: str, score_metric: TargetMetric):
    results_df = pd.read_csv(os.path.join(log_path, 'csv', 'training_results.csv'))

    acc_col = score_metric.results_csv_column
    best_record = results_df[results_df[acc_col] == results_df[acc_col].max()].to_dict('records')[0]
    top_accuracy = best_record[acc_col]

    top5_records = results_df.nlargest(5, columns=[acc_col])
    avg_top5_cells_acc = top5_records[acc_col].mean()

    return top_accuracy, avg_top5_cells_acc


def get_final_network_training_time_and_params(log_path: str):
    with open(os.path.join(log_path, 'final_model_training', 'debug.log'), 'r') as f:
        lines = f.read().splitlines()

    total_seconds, params = None, None
    for line in lines:
        match = re.search(r'Total training time \(without callbacks\): (\d+.\d+)', line)
        if match:
            total_seconds = float(match.group(1))

        match = re.search(r'Total params: (\d+(,\d+)*)', line)
        if match:
            params = int(match.group(1).replace(',', ''))

    return total_seconds, params


def get_final_network_macro_structure(log_path: str):
    with open(os.path.join(log_path, 'final_model_training', 'run.json'), 'r') as f:
        train_config = json.load(f)

    cnn_hp = train_config.training_hyperparameters
    arc_hp = train_config.architecture_hyperparameters

    return cnn_hp['filters'], arc_hp['motifs'], arc_hp['normal_cells_per_motif']


def get_final_network_cell_spec(log_path: str):
    with open(os.path.join(log_path, 'final_model_training', 'cell_spec.txt'), 'r') as f:
        cell_spec = f.readline()

    return CellSpecification.from_str(cell_spec)


def get_final_network_accuracy(log_path: str):
    with open(os.path.join(log_path, 'final_model_training', 'eval.txt'), 'r') as f:
        eval_txt = f.read()

    matches = re.findall(r"'Softmax_c(\d+)_accuracy': (\d+.\d+)", eval_txt)
    accuracies = {int(cell_index): float(acc) for cell_index, acc in matches}
    max_cell_index = max(accuracies.keys())

    return accuracies[max_cell_index]


def have_same_score_metric(infos: 'Sequence[Union[SearchInfo, FinalTrainingInfo]]'):
    first_el_score_metric = infos[0].score_metric
    return all([el.score_metric == first_el_score_metric for el in infos])


def write_search_infos_csv(save_path: str, infos: 'Sequence[SearchInfo]'):
    ''' Write a CSV file about the search results (i.e. num of sampled architectures, best accuracy found, total search time, etc...). '''
    if have_same_score_metric(infos):
        with open(save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(infos[0].csv_headers())
            writer.writerows([search_info.to_csv_row() for search_info in infos])
    else:
        print('Skipped search info CSV comparison due to different score metrics')


def write_final_training_infos_csv(save_path: str, infos: 'Sequence[FinalTrainingInfo]'):
    ''' Write a CSV file about the results of final trainings (i.e. cell specification, accuracy reached, macro structure, etc...). '''
    if have_same_score_metric(infos):
        with open(save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(infos[0].csv_headers())
            writer.writerows([ft_info.to_csv_row() for ft_info in infos])
    else:
        print('Skipped final training info CSV comparison due to different score metrics')
