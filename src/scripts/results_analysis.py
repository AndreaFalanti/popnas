import argparse
import csv
import json
import os
import re
from typing import Iterable

import pandas as pd

from utils.feature_utils import metrics_fields_dict
from utils.func_utils import parse_cell_structures, cell_spec_to_str


class SearchInfo:
    def __init__(self, dataset_name: str, num_networks: int, top_accuracy: float, mean5_top_accuracy: float, search_time: float):
        self.dataset_name = dataset_name
        self.num_networks = num_networks
        self.top_accuracy = top_accuracy
        self.mean5_top_accuracy = mean5_top_accuracy
        self.search_time_seconds = search_time
        self.search_time_hours = search_time / 3600
        self.search_time_str = from_seconds_to_time_str(search_time)

    @staticmethod
    def from_log_folder(log_folder: str):
        ''' Extract the search procedure info from the provided experiment output folder. '''
        search_time = get_total_search_time(log_folder)
        networks_count = get_sampled_networks_count(log_folder)
        ds_name = get_dataset_name(log_folder)
        best_acc, avg_top5_cells_acc = get_top_accuracies(log_folder)

        return SearchInfo(ds_name, networks_count, best_acc, avg_top5_cells_acc, search_time)

    def to_csv_row(self):
        return [self.dataset_name, self.num_networks, format(self.top_accuracy, '.3f'), format(self.mean5_top_accuracy, '.3f'), self.search_time_str]


class FinalTrainingInfo:
    def __init__(self, dataset_name: str, cell_spec: list, params: int, motifs: int, normal_cells_per_motif: int, filters: int,
                 accuracy: float, training_time: float):
        self.dataset_name = dataset_name
        self.cell_spec = cell_spec
        self.blocks = len(cell_spec)
        self.params = params
        self.motifs = motifs
        self.normal_cells_per_motif = normal_cells_per_motif
        self.filters = filters
        self.accuracy = accuracy
        self.training_time_seconds = training_time
        self.training_time_hours = training_time / 3600
        self.training_time_str = from_seconds_to_time_str(training_time)

    @staticmethod
    def from_log_folder(log_folder: str):
        ''' Extract the final training info from the provided experiment output folder. '''
        train_time, params = get_final_network_training_time_and_params(log_folder)
        f, m, n = get_final_network_macro_structure(log_folder)
        ds_name = get_dataset_name(log_folder)
        cell_spec = get_final_network_cell_spec(log_folder)
        acc = get_final_network_accuracy(log_folder)

        return FinalTrainingInfo(ds_name, cell_spec, params, m, n, f, acc, train_time)

    def to_csv_row(self):
        return [self.dataset_name, self.blocks, self.motifs, self.normal_cells_per_motif, self.filters, format(self.params / 1e6, '.2f'),
                format(self.accuracy, '.3f'), self.training_time_str, cell_spec_to_str(self.cell_spec)]


def from_seconds_to_time_str(total_seconds: float):
    total_seconds = int(total_seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds // 60) % 60
    seconds = total_seconds % 60

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
    with open(os.path.join(log_path, 'restore', 'run.json'), 'r') as f:
        run_config = json.load(f)

    return run_config['dataset']['name']


def get_top_accuracies(log_path: str):
    results_df = pd.read_csv(os.path.join(log_path, 'csv', 'training_results.csv'))

    acc_col = metrics_fields_dict['accuracy'].real_column
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

    cnn_hp = train_config['cnn_hp']
    arc_hp = train_config['architecture_parameters']

    return cnn_hp['filters'], arc_hp['motifs'], arc_hp['normal_cells_per_motif']


def get_final_network_cell_spec(log_path: str):
    with open(os.path.join(log_path, 'final_model_training', 'cell_spec.txt'), 'r') as f:
        cell_spec = f.readline()

    return parse_cell_structures([cell_spec])[0]


def get_final_network_accuracy(log_path: str):
    with open(os.path.join(log_path, 'final_model_training', 'eval.txt'), 'r') as f:
        eval_txt = f.read()

    matches = re.findall(r"'Softmax_c(\d+)_accuracy': (\d+.\d+)", eval_txt)
    accuracies = {int(cell_index): float(acc) for cell_index, acc in matches}
    max_cell_index = max(accuracies.keys())

    return accuracies[max_cell_index]


def write_search_infos(save_path: str, infos: 'Iterable[SearchInfo]'):
    ''' Write a CSV file about the search results (i.e. num of sampled architectures, best accuracy found, total search time, etc...). '''
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Dataset', '# Networks', 'Top Accuracy', 'Top-5 Cells Accuracy', 'Search Time'])
        writer.writerows([search_info.to_csv_row() for search_info in infos])


def write_final_training_infos(save_path: str, infos: 'Iterable[FinalTrainingInfo]'):
    ''' Write a CSV file about the results of final trainings (i.e. cell specification, accuracy reached, macro structure, etc...). '''
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Dataset', 'B', 'M', 'N', 'F', 'Params(M)', 'Accuracy', 'Training Time', 'Cell specification'])
        writer.writerows([ft_info.to_csv_row() for ft_info in infos])


def get_and_write_search_info(log_path: str):
    ''' Get search info and also write a single line CSV in the experiment folder. '''
    search_info = SearchInfo.from_log_folder(log_path)
    write_search_infos(os.path.join(log_path, 'csv', 'search_results.csv'), [search_info])

    return search_info


def get_and_write_final_training_info(log_path: str):
    ''' Get final training info and also write a single line CSV in the experiment folder. '''
    final_training_info = FinalTrainingInfo.from_log_folder(log_path)
    write_final_training_infos(os.path.join(log_path, 'final_model_training', 'results.csv'), [final_training_info])

    return final_training_info


def save_multiple_experiments_info(log_paths: 'list[str]', summary_files_save_location: str, summary_files_prefix: str = 'exp'):
    ''' Extract info from multiple experiments, writing a CSV per experiment and one summarizing all of them for comparisons. '''
    search_infos = [get_and_write_search_info(uni_path) for uni_path in log_paths]
    final_training_infos = [get_and_write_final_training_info(uni_path) for uni_path in log_paths]

    write_search_infos(os.path.join(summary_files_save_location, f'{summary_files_prefix}_search.csv'), search_infos)
    write_final_training_infos(os.path.join(summary_files_save_location, f'{summary_files_prefix}_final_training.csv'), final_training_infos)


def save_tsc_aggregated_info(summary_files_save_location: str):
    ''' Write csv aggregating univariate and multivariate results. '''
    uni_search_df = pd.read_csv(os.path.join(summary_files_save_location, 'univariate_search.csv'), index_col='Dataset')
    multi_search_df = pd.read_csv(os.path.join(summary_files_save_location, 'multivariate_search.csv'), index_col='Dataset')
    pd.concat([uni_search_df, multi_search_df]).to_csv(os.path.join(summary_files_save_location, 'all_search.csv'), float_format='%.3f')

    uni_final_df = pd.read_csv(os.path.join(summary_files_save_location, 'univariate_final_training.csv'), index_col='Dataset')
    multi_final_df = pd.read_csv(os.path.join(summary_files_save_location, 'multivariate_final_training.csv'), index_col='Dataset')
    pd.concat([uni_final_df, multi_final_df]).to_csv(os.path.join(summary_files_save_location, 'all_final_training.csv'), float_format='%.3f')


def execute(p: 'list[str]', tsca: bool = False):
    ''' Refer to argparse help for more information about these arguments. '''
    if tsca:
        if len(p) != 1:
            raise AttributeError('Provide only the TSC archives root folder when using the --tsca option')

        root_folder = p[0]
        univariate_paths = [f.path for f in os.scandir(os.path.join(root_folder, 'univariate')) if f.is_dir()]
        multivariate_paths = [f.path for f in os.scandir(os.path.join(root_folder, 'multivariate')) if f.is_dir()]

        save_multiple_experiments_info(univariate_paths, root_folder, summary_files_prefix='univariate')
        save_multiple_experiments_info(multivariate_paths, root_folder, summary_files_prefix='multivariate')
        save_tsc_aggregated_info(root_folder)
    else:
        save_multiple_experiments_info(p, p[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-p', metavar='FOLDER', nargs='+', type=str, help="log folders", required=True)
    parser.add_argument('--tsca', help="process a folder composed of multiple experiments on UCR/UEA archives", action="store_true")
    args = parser.parse_args()

    execute(**vars(args))
