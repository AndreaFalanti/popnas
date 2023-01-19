'''
Module responsible for writing various txt and csv files, saving the results extracted during POPNAS execution.
'''
import csv
import statistics

from tensorflow.keras import Model

import log_service
from models.results import BaseTrainingResults
from search_space import CellSpecification
from utils.model_estimate import ModelEstimate


def write_partitions_file(partition_dict: dict, save_path: str):
    ''' Write a txt file about partitions between cells of a neural network. '''
    lines = [f'{key}: {value:,} bytes' for key, value in partition_dict.items()]

    with open(save_path, 'w') as f:
        # writelines function usually adds \n automatically, but not in python...
        f.writelines(line + '\n' for line in lines)


def write_model_summary_file(cell_spec: CellSpecification, flops: float, model: Model, save_path: str):
    ''' Write Keras summary of the model structure, plus the total FLOPs of the model. '''
    with open(save_path, 'w') as f:
        # str casting is required since inputs are int
        f.write(f'Cell specification: {cell_spec}\n\n')
        model.summary(line_length=150, print_fn=lambda x: f.write(x + '\n'))
        f.write(f'\nFLOPs: {flops:,}')


def write_training_results_into_csv(train_res: BaseTrainingResults, exploration: bool = False):
    '''
    Append info about a single CNN training to the results csv file.
    '''
    with open(log_service.build_path('csv', 'training_results.csv'), mode='a+', newline='') as f:
        writer = csv.writer(f)

        # append mode, so if file handler is in position 0 it means is empty. In this case write the headers too
        if f.tell() == 0:
            writer.writerow(train_res.get_csv_headers() + ['exploration'])

        # trim cell structure from csv list and replace it with a valid string representation of it
        cell_structure_str = str(train_res.cell_spec)
        data = train_res.to_csv_list()[:-1] + [cell_structure_str, exploration]

        writer.writerow(data)


def write_overall_cnn_training_results(score_metric: str, blocks: int, train_results: 'list[BaseTrainingResults]'):
    ''' Write (average, max, min) of the training time and score achieved by all architectures trained during the experiment. '''
    def get_metric_aggregate_values(metric_values: 'list[float]'):
        return statistics.mean(metric_values), max(metric_values), min(metric_values)

    times, scores = [], []
    for train_res in train_results:
        times.append(train_res.training_time)
        scores.append(getattr(train_res, score_metric))

    with open(log_service.build_path('csv', 'training_overview.csv'), mode='a+', newline='') as f:
        writer = csv.writer(f)

        # append mode, so if file handler is in position 0 it means is empty. In this case write the headers too
        if f.tell() == 0:
            writer.writerow(['# blocks', 'avg training time(s)', 'max time', 'min time', 'avg val score', 'max score', 'min score'])

        avg_time, max_time, min_time = get_metric_aggregate_values(times)
        avg_acc, max_acc, min_acc = get_metric_aggregate_values(scores)

        writer.writerow([blocks, avg_time, max_time, min_time, avg_acc, max_acc, min_acc])


def write_specular_monoblock_times(smb_time_dict: 'dict[str, float]', save_path: str):
    ''' Save specular monoblock training times, so that on restore it is easier to restore the dynamic reindex map. '''
    with open(save_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows([v, k] for k, v in smb_time_dict.items())


def append_to_time_features_csv(cell_time_features: list):
    ''' Append time features extracted by a cell to the related csv file. '''
    with open(log_service.build_path('csv', 'training_time.csv'), mode='a+', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(cell_time_features)


def append_to_score_features_csv(cell_score_features: list):
    ''' Append score features extracted by a cell to the related csv file. '''
    with open(log_service.build_path('csv', 'training_score.csv'), mode='a+', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(cell_score_features)


def append_cell_spec_to_csv(cell_specs: list):
    ''' Append all cell specifications that will be trained in the next step (Pareto front + exploration Pareto front), in a csv file. '''
    with open(log_service.build_path('csv', 'children.csv'), mode='a+', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(cell_specs)


def save_predictions_to_csv(predictions: 'list[ModelEstimate]', filename: str):
    ''' Save a list of model estimation objects to a csv file. '''
    with open(log_service.build_path('csv', filename), mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(ModelEstimate.get_csv_headers())
        writer.writerows(map(lambda est: est.to_csv_array(), predictions))
