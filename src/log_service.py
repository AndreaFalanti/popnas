import glob
import logging
import os
import shutil
import sys
import tempfile
import time
from typing import Optional

import neptune
from neptune import management
from neptune.utils import stringify_unsupported
from neptune_tensorflow_keras import NeptuneCallback
from tensorflow.keras.callbacks import Callback

from search_space import CellSpecification
from utils.func_utils import from_seconds_to_hms

# Must be set using initialize_log_folders, check_log_folder or set_log_path
log_path = tempfile.mkdtemp()  # type: str
using_temp = True
neptune_project_name = None
neptune_project = None         # type: Optional[neptune.Project]


def set_log_path(path):
    '''
    Useful for scripts outside the main execution workflow, where the path folder already exists and
    extra folders creation is handled externally.
    '''
    global log_path
    global using_temp

    if using_temp:
        shutil.rmtree(log_path)
        using_temp = False

    log_path = path

    # NOTE: solves issue with Keras model.save duplicating logs. See also: https://github.com/keras-team/keras/issues/16732
    logging.root.addHandler(logging.NullHandler())


def get_log_folder_name():
    return os.path.split(log_path)[1]


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

    os.mkdir(os.path.join(log_path, 'csv'))  # create folder for csv files
    os.mkdir(os.path.join(log_path, 'best_model'))  # create folder for best model save
    os.mkdir(os.path.join(log_path, 'sampled_models'))  # create folder for saving models data
    os.mkdir(os.path.join(log_path, 'plots'))  # create folder for saving data plots
    os.mkdir(os.path.join(log_path, 'predictors'))  # create folder for saving predictors data
    os.mkdir(os.path.join(log_path, 'restore'))  # create folder for additional files used for restoring interrupted runs

    # additional folders for different plot formats
    # os.mkdir(os.path.join(log_path, 'plots', 'eps'))
    os.mkdir(os.path.join(log_path, 'plots', 'pdf'))


def restore_log_folder(folder_path: str):
    if not os.path.exists(folder_path):
        raise NotADirectoryError('Log directory not found')

    set_log_path(folder_path)


def get_logger(name, filename='debug.log'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Create handlers
    file_handler = logging.FileHandler(os.path.join(log_path, filename))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - [%(name)s:%(levelname)s] %(message)s'))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s'))

    # loggers with same name are actually the same logger, so they already have handlers in that case.
    # clear them and add the new ones. In this way, when running multiple experiments with e2e scripts, each experiments has its isolated log file.
    logger.handlers.clear()
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def create_critical_logger():
    logger = logging.getLogger('critical')
    file_handler = logging.FileHandler(os.path.join(log_path, 'critical.log'))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - [%(name)s:%(levelname)s] %(message)s'))
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(logging.Formatter('%(message)s'))

    # loggers with the same name are actually the same logger, so they already have handlers in that case.
    # clear them and add the new ones; in this way, when running multiple experiments with e2e scripts, each experiment has its isolated log file.
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# Taken from: https://stackoverflow.com/questions/6234405/logging-uncaught-exceptions-in-python
def make_exception_handler(logger):
    '''
    Closure to make an exception handler, given a logger.
    '''

    def handle_exception(exc_type, exc_value, exc_traceback):
        '''
        Handle uncaught exception logging, must be bound to sys.excepthook.
        '''
        # Avoid to log keyboard interrupts
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logger.error('Uncaught exception', exc_info=(exc_type, exc_value, exc_traceback))

    return handle_exception


def build_path(*args):
    return os.path.join(log_path, *args)


# region NEPTUNE_LOGGING
def _neptune_env_vars_missing():
    return os.environ.get('NEPTUNE_API_TOKEN') is None or os.environ.get('NEPTUNE_WORKSPACE') is None

def initialize_neptune_project():
    ''' Initialize a Neptune project if a valid API token is provided in the environment variables. '''
    if _neptune_env_vars_missing():
        print('WARNING: Neptune API key or workspace not provided, this run will not be logged on Neptune')
        return

    run_name = get_log_folder_name()

    global neptune_project_name
    neptune_project_name = management.create_project(
        name=f'popnas-{run_name}',
        workspace=os.environ.get('NEPTUNE_WORKSPACE'),
        visibility=management.ProjectVisibility.PRIVATE
    )

    global neptune_project
    neptune_project = neptune.init_project(neptune_project_name)


def restore_neptune_project(run_name: str = None):
    '''
    Restore connection to a Neptune project if a valid API token is provided in the environment variables.
    Necessary to continue logging on Neptune when restoring an interrupted POPNAS experiment.
    '''
    if _neptune_env_vars_missing():
        print('WARNING: Neptune API key or workspace not provided, this run will not be logged on Neptune')
        return

    if run_name is None:
        run_name = get_log_folder_name()

    global neptune_project_name
    neptune_project_name = f'popnas-{run_name}'

    global neptune_project
    neptune_project = neptune.init_project(neptune_project_name)


def write_config_into_neptune(config_dict: dict, config_path: str):
    ''' Save config info into the Neptune project, if instantiated. '''
    if neptune_project is None:
        return
    
    neptune_project['popnas_config'] = stringify_unsupported(config_dict)
    neptune_project['popnas_config_json'].upload(config_path)


def save_summary_files_and_metadata_into_neptune(total_search_time: int, num_networks_trained: int):
    ''' Save search summary info and files into the Neptune project, if instantiated. '''
    if neptune_project is None:
        return

    # save summary info into Neptune project, if instantiated
    hours, minutes, seconds = from_seconds_to_hms(total_search_time)
    neptune_project['search_time'] = f'{hours} hours {minutes} minutes {seconds} seconds'
    neptune_project['networks_trained'] = num_networks_trained

    # upload plot images
    plot_images = [f for f in os.scandir(build_path('plots')) if f.is_file()]  # type: list[os.DirEntry]
    for plot_entry in plot_images:
        neptune_project[f'plots/{plot_entry.name}'].upload(plot_entry.path)

    # upload csv files
    csv_files = [f for f in os.scandir(build_path('csv')) if f.is_file()]  # type: list[os.DirEntry]
    for csv_entry in csv_files:
        neptune_project[f'csv/{csv_entry.name}'].upload(csv_entry.path)

    # upload logs
    neptune_project['logs'].upload(build_path('debug.log'))
    neptune_project['errors'].upload(build_path('critical.log'))

    neptune_project.sync()


def generate_neptune_run(run_name: str, tags: 'list[str]', cell_spec: 'CellSpecification', callbacks: 'list[Callback]'):
    '''
    Initialize a Neptune run under the experiment project, to store data related to a particular model training session.
    If there is no Neptune project (optional credentials have not been provided), then the Neptune run is not created.
    '''
    if neptune_project_name is None:
        return None, callbacks

    # create Neptune run and add the related callback
    neptune_run = neptune.init_run(project=neptune_project_name, name=run_name,
                                   source_files=[], tags=tags)

    neptune_run['cell_specification'] = str(cell_spec)
    callbacks.append(NeptuneCallback(run=neptune_run, log_model_diagram=True))

    return neptune_run, callbacks


def finalize_neptune_run(neptune_run: neptune.Run, params: int, training_time: int):
    ''' Write conclusive info about a model training session and close the Neptune run. '''
    if neptune_run is None:
        return

    neptune_run['training_time(seconds)'] = training_time
    neptune_run['params'] = params

    neptune_run.stop()


def save_macro_hp_in_neptune_run(neptune_run: neptune.Run, motifs: int, normal_cells_per_motif: int, filters: int):
    ''' Save macro hyperparameters in the Neptune run. '''
    if neptune_run is None:
        return

    neptune_run['motifs'] = motifs
    neptune_run['normal_cells_per_motif'] = normal_cells_per_motif
    neptune_run['filters'] = filters


def save_model_in_neptune_registry(tf_model_path: str, onnx_path: str, cell_spec: 'CellSpecification', params: int,
                                   motifs: int, normal_cells_per_motif: int, filters: int):
    ''' Save the model files into the Neptune model registry of the current project. '''
    if neptune_project is None:
        return

    try:
        neptune_model = neptune.init_model(project=neptune_project_name, key='DEPLOYMENT', name='Final training model')
    except:
        neptune_models_df = neptune_project.fetch_models_table().to_pandas()
        model_id = neptune_models_df["sys/id"].values[0]
        neptune_model = neptune.init_model(project=neptune_project_name, with_id=model_id)

    model_id = neptune_model['sys/id'].fetch()
    model_version = neptune.init_model_version(project=neptune_project_name, model=model_id)

    # save TF model
    model_version['tf-model/saved_model'].upload(os.path.join(tf_model_path, 'saved_model.pb'))
    model_version['tf-model/keras_metadata'].upload(os.path.join(tf_model_path, 'keras_metadata.pb'))

    for file_path in glob.glob(f'{tf_model_path}/variables/*'):
        file_name = os.path.split(file_path)[1]
        model_version[f'tf-model/variables/{file_name}'].upload(file_path)

    # save ONNX model
    model_version['onnx'].upload(onnx_path)

    # save metadata
    model_version['cell_spec'] = str(cell_spec)
    model_version['params'] = params
    macro = {
        'motifs': motifs,
        'normal': normal_cells_per_motif,
        'filters': filters
    }
    model_version['macro_architecture'] = macro

    model_version.stop()
# endregion
