import argparse
import json
import os.path
import sys

import tensorflow as tf

import log_service
from train import Train


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-r', metavar='RESTORE_FOLDER', type=str, help="log folder to restore", default=None)
    parser.add_argument('-j', metavar='JSON_PATH', type=str, help="path to config json with run parameters", default=None)
    parser.add_argument('--cpu', help="use CPU instead of GPU", action="store_true")
    parser.add_argument('--pnas', help="run in PNAS mode (no regressor, only LSTM controller)", action="store_true")
    args = parser.parse_args()

    if tf.test.gpu_device_name():
        print('GPU found')
    else:
        print("No GPU found")

    device_list = tf.config.list_physical_devices()
    print(device_list)

    gpu_devices = tf.config.list_physical_devices('GPU')

    # DEBUG: To find out which devices your operations and tensors are assigned to
    # tf.debugging.set_log_device_placement(True)

    if args.cpu:
        # remove GPUs from visible devices, using only CPUs
        tf.config.set_visible_devices([], 'GPU')
        print("Using CPU devices only")
    elif not args.cpu and len(gpu_devices) == 0:
        sys.exit('GPU is not available for execution, run with --cpu flag or troubleshot the issue in case a GPU is actually present in the device')

    # create folder structure for log files or reuse previous logs to continue execution
    if args.r is not None:
        # log_service.check_log_folder(args.r)
        raise NotImplementedError('Restore functionality is actually not supported')
    else:
        log_service.initialize_log_folders()

        json_path = os.path.join('configs', 'run.json') if args.j is None else args.j
        with open(json_path, 'r') as f:
            run_config = json.load(f)

    run_config['pnas_mode'] = args.pnas
    run_config['use_cpu'] = args.cpu

    # Handle uncaught exception in a special log file
    sys.excepthook = log_service.make_exception_handler(log_service.create_critical_logger())

    # print info about the command line arguments provided. Optional ones will list their default value.
    logger = log_service.get_logger(__name__)
    logger.info('%s', '*' * 31 + ' COMMAND LINE ARGUMENTS (WITH DEFAULTS) ' + '*' * 31)
    for arg in vars(args):
        logger.info('%s: %s', arg, getattr(args, arg))
    logger.info('%s', '*' * 101)

    run = Train(run_config)
    run.process()


if __name__ == '__main__':
    main()
