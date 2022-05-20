import argparse
import json
import os.path
import sys

import tensorflow as tf

import log_service
from train import Train


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-r', metavar='RESTORE_FOLDER', type=str, help='path of log folder to restore (timestamp-named folder)', default=None)
    parser.add_argument('-j', metavar='JSON_PATH', type=str, help='path to config json with run parameters', default=None)
    parser.add_argument('--name', metavar='RUN_NAME', type=str, help='name used for log folder', default=None)
    args = parser.parse_args()

    if tf.test.gpu_device_name():
        print('GPU found')
    else:
        print('No GPU found')

    device_list = tf.config.list_physical_devices()
    print(device_list)

    gpu_devices = tf.config.list_physical_devices('GPU')

    # create folder structure for log files or reuse previous logs to continue execution
    if args.r is not None:
        log_service.check_log_folder(args.r)
        # load the exact configuration in which the run was started (also setting CPU/GPU properly to maintain training time consistency)
        with open(log_service.build_path('restore', 'run.json'), 'r') as f:
            run_config = json.load(f)

        gpu_msg = 'Restored run used GPU, to make training time consistent you have to continue the run on a GPU-powered device'
    else:
        json_path = os.path.join('configs', 'run.json') if args.j is None else args.j
        with open(json_path, 'r') as f:
            run_config = json.load(f)

        gpu_msg = 'GPU is not available for execution, run with --cpu flag or troubleshot the issue in case a GPU is actually present in the device'

    # DEBUG: To find out which devices your operations and tensors are assigned to
    # tf.debugging.set_log_device_placement(True)

    if run_config['use_cpu']:
        # remove GPUs from visible devices, using only CPUs
        tf.config.set_visible_devices([], 'GPU')
        print('Using CPU devices only')
    elif not run_config['use_cpu'] and len(gpu_devices) == 0:
        sys.exit(gpu_msg)

    # initialize folders after CPU/GPU check, just to avoid making folders when run is faulty
    if args.r is None:
        log_service.initialize_log_folders(args.name)
        # copy config (with args override) for possible run restore
        with open(log_service.build_path('restore', 'run.json'), 'w') as f:
            json.dump(run_config, f, indent=4)

    # Handle uncaught exception in a special log file
    sys.excepthook = log_service.make_exception_handler(log_service.create_critical_logger())

    # print info about the command line arguments provided. Optional ones will list their default value.
    # logger = log_service.get_logger(__name__)
    # logger.info('%s', '*' * 31 + ' COMMAND LINE ARGUMENTS (WITH DEFAULTS) ' + '*' * 31)
    # for arg in vars(args):
    #     logger.info('%s: %s', arg, getattr(args, arg))
    # logger.info('%s', '*' * 101)

    run = Train(run_config)
    run.process()


if __name__ == '__main__':
    main()
