import argparse
import sys

import log_service
from train import Train

import tensorflow as tf


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-b', metavar='BLOCKS', type=int, help="maximum number of blocks a cell can contain", required=True)
    parser.add_argument('-k', metavar='CHILDREN', type=int, help="number of top-K cells to expand at each iteration", required=True)
    parser.add_argument('-c', metavar='CHECKPOINT', type=int, help="checkpoint number of blocks to restore", default=1)
    parser.add_argument('-d', metavar='DATASET', type=str, help="python file to use as dataset", default="cifar10")
    parser.add_argument('-s', metavar='NUM_DATASETS', type=int, help="how many times a child network has to be trained", default=1)
    parser.add_argument('-e', metavar='EPOCHS', type=int, help="number of epochs each child network has to be trained", default=20)
    parser.add_argument('-z', metavar='BATCH_SIZE', type=int, help="batch size dimension of the dataset", default=128)
    parser.add_argument('-l', metavar='LEARNING_RATE', type=float, help="learning rate of the child networks", default=0.01)
    parser.add_argument('-r', '--restore', help="restore a previous run", action="store_true")
    parser.add_argument('-t', metavar='FOLDER', type=str, help="log folder to restore", default="")
    parser.add_argument('-f', metavar='FILTERS', type=int, help="initial number of filters", default=24)
    parser.add_argument('-m', metavar='CELL_STACKS', type=int, help="number of cell stacks to use when building the CNNs", default=3)
    parser.add_argument('-n', metavar='NORMAL_CELLS', type=int, help="number of normal cells (stride 1) to use in a cell stack", default=2)
    parser.add_argument('-wr', metavar='WEIGHT_REG', type=float, help="L2 weight regularization factor, not applied if not specified", default=None)
    parser.add_argument('--cpu', help="use CPU instead of GPU", action="store_true")
    parser.add_argument('--abc', help="concat all blocks output in a cell output, otherwise use unused only", action="store_true")
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
    if args.restore:
        log_service.check_log_folder(args.t)
    else:
        log_service.initialize_log_folders()

    # Handle uncaught exception in a special log file
    sys.excepthook = log_service.make_exception_handler(log_service.create_critical_logger())

    run = Train(args.b, args.k,
                dataset=args.d, sets=args.s,
                epochs=args.e, batch_size=args.z, learning_rate=args.l, filters=args.f, weight_reg=args.wr,
                cell_stacks=args.m, normal_cells_per_stack=args.n,
                all_blocks_concat=args.abc, pnas_mode=args.pnas,
                checkpoint=args.c, restore=args.restore)

    run.process()


if __name__ == '__main__':
    main()
