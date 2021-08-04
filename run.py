import argparse

import train

import log_service
import sys

import tensorflow as tf


def main():

    parser = argparse.ArgumentParser(description = "")
    parser.add_argument('-b', metavar=('BLOCKS'), type=int, help = "maximum number of blocks a cell can contain", required = True)
    parser.add_argument('-k', metavar=('CHILDREN'), type=int, help = "number of top-K cells to expand at each iteration", required = True)
    parser.add_argument('-c', metavar=('CHECKPOINT'), type=int, help = "checkpoint number of blocks to restore", default = 1)
    parser.add_argument('-d', metavar=('DATASET'), type=str, help = "python file to use as dataset", default = "cifar10")
    parser.add_argument('-s', metavar=('NUM_DATASETS'), type=int, help = "how many times a child network has to be trained", default = 1)
    parser.add_argument('-e', metavar=('EPOCHS'), type=int, help = "number of epochs each child network has to be trained", default = 20)
    parser.add_argument('-z', metavar=('BATCH_SIZE'), type=int, help = "batch size dimension of the dataset", default = 128)
    parser.add_argument('-l', metavar=('LEARNING_RATE'), type=float, help = "learning rate of the child networks", default = 0.01)
    parser.add_argument('-r', '--restore', help = "restore a previous run", action = "store_true")
    parser.add_argument('-t', metavar=('FOLDER'), type=str, help = "log folder to restore", default = "")
    parser.add_argument('--cpu', help="use CPU instead of GPU", action="store_true")
    parser.add_argument('--abc', help="concat all blocks output in a cell output, otherwise use unused only", action="store_true")
    args = parser.parse_args()

    if tf.test.gpu_device_name():
        print('GPU found')
    else:
        print("No GPU found")

    device_list = tf.config.experimental.get_visible_devices()
    print(device_list)
    
    if (not args.cpu and not tf.test.is_gpu_available()):
        sys.exit('GPU is not available for execution, run with --cpu flag or troubleshot the issue in case a GPU is actually present in the device')

    # create folder structure for log files or reuse previous logs to continue execution
    if (args.restore == True):
        log_service.check_log_folder(args.t)
    else:
        log_service.initialize_log_folders()

    # Handle uncaught exception in a special log file
    sys.excepthook = log_service.make_exception_handler(log_service.create_critical_logger())

    run = train.Train(args.b, args.k, checkpoint=args.c,
                      dataset=args.d, sets=args.s, epochs=args.e, batchsize=args.z,
                      learning_rate=args.l, restore=args.restore, cpu=args.cpu, all_blocks_concat=args.abc)

    run.process()

if __name__ == '__main__':
    main()
