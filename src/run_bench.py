import argparse
import dataclasses
import json
import os.path
import sys

import numpy as np
import pandas as pd
from dacite import from_dict

import log_service
from benchmarking import NATSbench
from popnas import Popnas
from search_space import CellSpecification
from utils.config_dataclasses import RunConfig
from utils.config_utils import validate_config_json
from utils.func_utils import create_empty_folder
from utils.nn_utils import initialize_train_strategy
from utils.rstr import rstr


def generate_popnas_bench_config(dataset_name: str, bench_path: str):
    return {
        'search_space': {
            'blocks': 2,
            'lookback_depth': 1,
            'operators': [
                'identity',
                '1x1 conv',
                '3x3 conv',
                '3x3 avgpool'
            ]
        },
        'search_strategy': {
            'max_children': 128,
            'max_exploration_children': 0,
            'score_metric': 'accuracy',
            'additional_pareto_objectives': ['time', 'params']
        },
        'training_hyperparameters': {
            'epochs': 200,
            'learning_rate': 0.01,
            'weight_decay': 5e-4,
            'use_adamW': True,
            'drop_path': 0.0,
            'softmax_dropout': 0.0,
            'optimizer': {
                'type': 'adamW',
                'scheduler': 'cdr: 3 period'
            }
        },
        'architecture_hyperparameters': {
            'filters': 24,
            'motifs': 3,
            'normal_cells_per_motif': 5,
            'block_join_operator': 'add',
            'lookback_reshape': True,
            'concat_only_unused_blocks': True,
            "residual_cells": False,
            "se_cell_output": False,
            'multi_output': False
        },
        'dataset': {
            'type': 'image_classification',
            'name': dataset_name,
            'path': bench_path,
            # NOTE: other parameters are not necessary in bench, but kept to avoid potential problems
            'classes_count': 10,
            'batch_size': 128,
            'inference_batch_size': 16,
            'validation_size': 0.1,
            'cache': True,
            'folds': 1,
            'samples': None,
            'balance_class_losses': False,
            'data_augmentation': {
                'enabled': True,
                'use_cutout': False
            }
        },
        'others': {
            'accuracy_predictor_ensemble_units': 5,
            'predictions_batch_size': 1024,
            'save_children_weights': False,
            'save_children_models': False,
            'pnas_mode': False,
            'train_strategy': 'GPU'
        }
    }


def get_best_cell_spec(log_folder_path: str, metric: str = 'best val accuracy'):
    training_results_csv_path = os.path.join(log_folder_path, 'csv', 'training_results.csv')
    df = pd.read_csv(training_results_csv_path)
    best_acc_row = df.loc[df[metric].idxmax()]
    cell_spec = CellSpecification.from_str(best_acc_row['cell structure'])

    return cell_spec, best_acc_row[metric]


def evaluate_best_cell_on_test(config: RunConfig, run_folder: str):
    # Create the API instance for the topology search space in NATS
    bench = NATSbench(config.dataset.path)

    best_cell_spec, best_val = get_best_cell_spec(run_folder)
    test_acc = bench.simulate_testing_on_nas_bench_201(best_cell_spec, config.dataset.name)

    with open(os.path.join(run_folder, 'bench-test.txt'), 'w') as f:
        f.write(f'Cell: {rstr(best_cell_spec)}\n')
        f.write(f'val: {best_val}, test: {test_acc}')

    return best_val, test_acc


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-p', metavar='BENCH_PATH', type=str, help='path to NAS-Bench-201 files', default=None, required=True)
    parser.add_argument('-n', metavar='N_RUNS', type=int, help='number of runs to execute on each dataset', default=3)
    parser.add_argument('--name', metavar='RUN_NAME', type=str, help='name used for log folder', default=None)
    args = parser.parse_args()

    # will contain all runs executed
    main_folder = args.name if args.name is not None else 'nas-bench-201'
    create_empty_folder(os.path.join('logs', main_folder))

    supported_datasets = ('cifar10', 'cifar100', 'ImageNet16-120')
    for dataset in supported_datasets:
        run_config_dict = generate_popnas_bench_config(dataset, args.p)
        run_config = from_dict(data_class=RunConfig, data=run_config_dict)
        # check that the config is correct
        validate_config_json(run_config)

        train_strategy = initialize_train_strategy(run_config.others.train_strategy, run_config.others.use_mixed_precision)

        val_accuracies, test_accuracies = [], []

        for i in range(args.n):
            run_folder_path = os.path.join(main_folder, f'{dataset}_{i}')
            log_service.initialize_log_folders(run_folder_path)

            # copy config (with args override) for possible run restore
            with open(log_service.build_path('restore', 'run.json'), 'w') as f:
                json.dump(dataclasses.asdict(run_config), f, indent=4)

            # handle uncaught exception in a special log file
            sys.excepthook = log_service.make_exception_handler(log_service.create_critical_logger())

            popnas = Popnas(run_config, train_strategy, benchmarking=True)
            popnas.start()

            # perform final evaluation on best cell found
            val_acc, test_acc = evaluate_best_cell_on_test(run_config, os.path.join('logs', run_folder_path))
            val_accuracies.append(val_acc * 100)
            test_accuracies.append(test_acc * 100)

        val_mean, test_mean = np.mean(val_accuracies), np.mean(test_accuracies)
        val_variance, test_variance = np.var(val_accuracies), np.var(test_accuracies)
        with open(os.path.join('logs', main_folder, f'{dataset}.txt'), 'w') as f:
            f.write(f'DATASET: {dataset}\n')
            f.write(f'Validation accuracies: {val_accuracies} -> {val_mean:0.2f}+-({val_variance:0.2f})\n')
            f.write(f'Test accuracies: {test_accuracies} -> {test_mean:0.2f}+-({test_variance:0.2f})\n')


if __name__ == '__main__':
    main()
