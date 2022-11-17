import logging
import os
import sys
import time

# Must be set using initialize_log_folders, check_log_folder or set_log_path
log_path = None  # type: str


def set_log_path(path):
    '''
    Useful for scripts outside the main execution workflow, where the path folder already exists and
    extra folders creation is handled externally.
    '''
    global log_path
    log_path = path

    # NOTE: solves issue with Keras model.save duplicating logs. See also: https://github.com/keras-team/keras/issues/16732
    logging.root.addHandler(logging.NullHandler())


def initialize_log_folders(folder_name: str = None):
    '''
    Used in POPNAS algorithm, initialize logs folder and subfolders.
    '''
    if not os.path.exists('logs/'):
        os.mkdir('logs/')

    # create timestamp subfolder if name is None, otherwise use the given name
    log_folder = time.strftime('%Y-%m-%d-%H-%M-%S') if folder_name is None else folder_name

    set_log_path(os.path.join('logs', log_folder))

    try:
        os.makedirs(log_path, exist_ok=False)
    except FileExistsError:
        raise AttributeError('The provided log folder name already exists, use another name to avoid conflicts')

    os.mkdir(os.path.join(log_path, 'csv'))  # create .csv path
    os.mkdir(os.path.join(log_path, 'best_model'))  # create folder for best model save
    os.mkdir(os.path.join(log_path, 'tensorboard_cnn'))  # create folder for saving tensorboard logs
    os.mkdir(os.path.join(log_path, 'plots'))  # create folder for saving data plots
    os.mkdir(os.path.join(log_path, 'predictors'))  # create folder for saving predictors' outputs
    os.mkdir(os.path.join(log_path, 'restore'))  # create folder for additional files used in restore mode

    # additional folders for different plot formats
    # os.mkdir(os.path.join(log_path, 'plots', 'eps'))
    os.mkdir(os.path.join(log_path, 'plots', 'pdf'))


def check_log_folder(folder_path: str):
    if not os.path.exists(folder_path):
        raise NotADirectoryError('Log directory not found')

    global log_path
    log_path = folder_path


def get_logger(name, filename='debug.log'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Create handlers
    file_handler = logging.FileHandler(os.path.join(log_path, filename))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - [%(name)s:%(levelname)s] %(message)s"))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    # loggers with same name are actually the same logger, so they already have handlers in that case.
    # clear them and add the new ones. In this way, when running multiple experiments with e2e scripts, each experiments has its isolated log file.
    logger.handlers.clear()
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def create_critical_logger():
    logger = logging.getLogger(__name__)
    file_handler = logging.FileHandler(os.path.join(log_path, 'critical.log'))
    file_handler.setFormatter(logging.Formatter("%(asctime)s - [%(name)s:%(levelname)s] %(message)s"))
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

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
