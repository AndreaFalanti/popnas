import argparse
import json
import sys

import log_service
from popnas import Popnas
from utils.config_validator import validate_config_json
from utils.nn_utils import initialize_train_strategy


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
        # TODO: params objective currently not supported, but could be fetched too from NAS-Bench-201
        'search_strategy': {
            'max_children': 128,
            'max_exploration_children': 0,
            'score_metric': 'accuracy',
            'additional_pareto_objectives': ['time']
        },
        'cnn_hp': {
            'epochs': 200,
            'learning_rate': 0.01,
            'filters': 24,
            'weight_reg': 5e-4,
            'use_adamW': True,
            'drop_path_prob': 0.0,
            'cosine_decay_restart': {
                'enabled': True,
                'period_in_epochs': 3,
                't_mul': 2.0,
                'm_mul': 1.0,
                'alpha': 0.0
            },
            'softmax_dropout': 0.0
        },
        'architecture_parameters': {
            'motifs': 3,
            'normal_cells_per_motif': 5,
            'block_join_operator': 'add',
            'lookback_reshape': True,
            'concat_only_unused_blocks': True,
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
            'resize': {
                'enabled': False,
                'width': 224,
                'height': 224
            },
            'data_augmentation': {
                'enabled': True,
                'perform_on_gpu': False
            }
        },
        'others': {
            'accuracy_predictor_ensemble_units': 5,
            'predictions_batch_size': 1024,
            'save_children_weights': False,
            'save_children_as_onnx': False,
            'pnas_mode': False,
            'train_strategy': 'GPU'
        }
    }


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', metavar='DATASET', type=str, help='name of NAS-Bench-201 dataset', default=None, required=True)
    parser.add_argument('-p', metavar='BENCH_PATH', type=str, help='path to NAS-Bench-201 files', default=None, required=True)
    parser.add_argument('--name', metavar='RUN_NAME', type=str, help='name used for log folder', default=None)
    args = parser.parse_args()

    log_service.initialize_log_folders(args.name)

    run_config = generate_popnas_bench_config(args.d, args.p)

    # copy config (with args override) for possible run restore
    with open(log_service.build_path('restore', 'run.json'), 'w') as f:
        json.dump(run_config, f, indent=4)

    # Handle uncaught exception in a special log file
    sys.excepthook = log_service.make_exception_handler(log_service.create_critical_logger())

    # check that the config is correct
    validate_config_json(run_config)

    # DEBUG: To find out which devices your operations and tensors are assigned to
    # tf.debugging.set_log_device_placement(True)

    train_strategy = initialize_train_strategy(run_config['others']['train_strategy'])

    popnas = Popnas(run_config, train_strategy, benchmarking=True)
    popnas.start()


if __name__ == '__main__':
    main()
