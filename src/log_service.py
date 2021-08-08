import logging
import sys
import os
import time
from pathlib import Path

# Must be set using initialize_log_folders or check_log_folder
log_path = None


def initialize_log_folders():
    if not os.path.exists('logs/'):
        os.mkdir('logs/')

    global log_path
    # create timestamp subfolder and set the global log path variable
    timestr = time.strftime('%Y-%m-%d-%H-%M-%S')  # get time for logs folder
    log_path = os.path.join('logs', timestr)
    os.mkdir(log_path)

    # TODO: folders not listed here are created through "tf.contrib.summary.create_file_writer" function,
    #  but their content is very strange, investigate
    os.mkdir(os.path.join(log_path, 'csv'))  # create .csv path
    os.mkdir(os.path.join(log_path, 'ini'))  # create .ini folder
    os.mkdir(os.path.join(log_path, 'weights'))  # create weights folder
    os.mkdir(os.path.join(log_path, 'best_model'))  # create folder for best model save


def check_log_folder(timestamp):
    if not os.path.exists(os.path.join('logs', timestamp)):
        raise Exception('Log directory not found')

    global log_path
    log_path = os.path.join('logs', timestamp)


def get_logger(name):
    logger = logging.getLogger(name)

    # Create handlers
    file_handler = logging.FileHandler(os.path.join(log_path, 'debug.log'))
    file_handler.setFormatter(logging.Formatter("%(asctime)s - [%(name)s:%(levelname)s] %(message)s"))
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    logging.basicConfig(
        level=logging.INFO,
        #format="%(asctime)s - [%(name)s:%(levelname)s] %(message)s",
        handlers=[
            file_handler,
            console_handler
        ]
    )
    return logger


def create_critical_logger():
    logger = logging.getLogger(__name__)
    fHandler = logging.FileHandler(os.path.join(log_path, 'critical.log'))
    fHandler.setFormatter(logging.Formatter("%(asctime)s - [%(name)s:%(levelname)s] %(message)s"))
    logger.addHandler(fHandler)

    return logger

# Taken from: https://stackoverflow.com/questions/6234405/logging-uncaught-exceptions-in-python


def make_exception_handler(logger):
    """
    Closure to make an exception handler, given a logger.
    """
    def handle_exception(exc_type, exc_value, exc_traceback):
        """
        Handle uncaught exception logging, must be bound to sys.excepthook.
        """
        # Avoid to log keyboard interrupts
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    return handle_exception


def build_path(*args):
    return os.path.join(log_path, *args)
