import argparse
import json
import os.path
import sys

import log_service
from train import Train
from utils.nn_utils import initialize_train_strategy


def validate_pareto_objectives(objectives: 'list[str]'):
    # TODO: config structure check should be done in entirety in a single place, think about it in the future
    valid_pareto_objectives = ['accuracy', 'time', 'params']

    if len(objectives) <= 1:
        raise ValueError('You must provide at least two objectives to optimize (see config search_strategy.pareto_objectives)')
    # TODO: actually we need to have at least one metric for prediction quality (accuracy, f1score, ecc..) to have meaningful results.
    #  Other metrics should be included in future.
    if 'accuracy' not in objectives:
        raise ValueError('Accuracy must always be included in Pareto objectives')

    for obj in objectives:
        if obj not in valid_pareto_objectives:
            raise ValueError(f'Invalid Pareto objective provided ({obj})')


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-r', metavar='RESTORE_FOLDER', type=str, help='path of log folder to restore (timestamp-named folder)', default=None)
    parser.add_argument('-j', metavar='JSON_PATH', type=str, help='path to config json with run parameters', default=None)
    parser.add_argument('--name', metavar='RUN_NAME', type=str, help='name used for log folder', default=None)
    args = parser.parse_args()

    # create folder structure for log files or reuse previous logs to continue execution
    if args.r is not None:
        log_service.check_log_folder(args.r)
        # load the exact configuration in which the run was started (also setting CPU/GPU properly to maintain training time consistency)
        with open(log_service.build_path('restore', 'run.json'), 'r') as f:
            run_config = json.load(f)
    else:
        json_path = os.path.join('configs', 'run.json') if args.j is None else args.j
        with open(json_path, 'r') as f:
            run_config = json.load(f)

    # DEBUG: To find out which devices your operations and tensors are assigned to
    # tf.debugging.set_log_device_placement(True)

    train_strategy = initialize_train_strategy(run_config['others']['train_strategy'])

    validate_pareto_objectives(run_config['search_strategy']['pareto_objectives'])

    # initialize folders after CPU/GPU check, just to avoid making folders when run is faulty
    if args.r is None:
        log_service.initialize_log_folders(args.name)
        # copy config (with args override) for possible run restore
        with open(log_service.build_path('restore', 'run.json'), 'w') as f:
            json.dump(run_config, f, indent=4)

    # Handle uncaught exception in a special log file
    sys.excepthook = log_service.make_exception_handler(log_service.create_critical_logger())

    run = Train(run_config, train_strategy)
    run.process()


if __name__ == '__main__':
    main()
