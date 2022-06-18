import argparse
import json
import os.path
import sys

import tensorflow as tf

import log_service
from train import Train


def initialize_train_strategy(config_strategy: str, device_inconsistency_msg: str):
    # debug available devices
    device_list = tf.config.list_physical_devices()
    print(device_list)

    gpu_devices = tf.config.list_physical_devices('GPU')
    tpu_devices = tf.config.list_physical_devices('TPU')

    # if tf.test.gpu_device_name():
    #     print('GPU found')
    # else:
    #     print('No GPU found')

    # TODO: add multi-GPU
    # generate the train strategy. currently supported values: ['CPU', 'GPU', 'TPU']
    if config_strategy == 'CPU':
        # remove GPUs from visible devices, using only CPUs
        tf.config.set_visible_devices([], 'GPU')
        tf.config.set_visible_devices([], 'TPU')
        print('Using CPU devices only')
        # default strategy
        train_strategy = tf.distribute.get_strategy()
    elif config_strategy == 'GPU':
        if len(gpu_devices) == 0:
            sys.exit(device_inconsistency_msg)
        # default strategy also for single GPU
        train_strategy = tf.distribute.get_strategy()
    elif config_strategy == 'TPU':
        if len(tpu_devices) == 0:
            sys.exit(device_inconsistency_msg)
        cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
        tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
        train_strategy = tf.distribute.TPUStrategy(cluster_resolver)
    else:
        sys.exit('Train strategy provided in configuration file is invalid')

    return train_strategy


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

        missing_device_msg = f'Restored run used {run_config["train_strategy"]}, to make training time consistent' \
                  f' you have to continue the run on a {run_config["train_strategy"]}-powered device.'
    else:
        json_path = os.path.join('configs', 'run.json') if args.j is None else args.j
        with open(json_path, 'r') as f:
            run_config = json.load(f)

        missing_device_msg = f'{run_config["train_strategy"]} is not available for execution, run with a different train strategy' \
                             f' or troubleshot the issue in case a {run_config["train_strategy"]} is actually present in the device.'

    # DEBUG: To find out which devices your operations and tensors are assigned to
    # tf.debugging.set_log_device_placement(True)

    train_strategy = initialize_train_strategy(run_config['train_strategy'], missing_device_msg)

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
