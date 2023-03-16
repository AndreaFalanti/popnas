import functools
import operator
import os
import shutil
from configparser import ConfigParser
from multiprocessing import Process
from typing import Iterable, Callable

import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error


def to_int_tuple(str_tuple: Iterable[str]):
    '''
    Cast each str element of a tuple to int and return it.
    '''
    return tuple(map(int, str_tuple))


def list_flatten(nested_l: 'list[Iterable]'):
    return [el for iterable_el in nested_l for el in iterable_el]


def to_list_of_tuples(seq, chunk_size):
    return list(tuple(seq[pos:(pos + chunk_size)]) for pos in range(0, len(seq), chunk_size))


def clamp(n: float, lower_bound: float, upper_bound: float):
    return max(lower_bound, min(n, upper_bound))


def elementwise_mult(a: Iterable[float], b: Iterable[float]):
    return [el_a * el_b for el_a, el_b in zip(a, b)]


def compute_spearman_rank_correlation_coefficient(y_true: 'list[float]', y_pred: 'list[float]'):
    '''
    Spearman rank correlation coefficient, given two lists.
    '''
    df = pd.DataFrame.from_dict({'true': y_true, 'est': y_pred})
    return df['true'].corr(df['est'], method='spearman')


def compute_spearman_rank_correlation_coefficient_from_df(x_col: pd.Series, y_col: pd.Series):
    '''
    Spearman rank correlation coefficient, computed on given Pandas dataframe.
    '''
    return x_col.corr(y_col, method='spearman')


def compute_mape(y_true: Iterable[float], y_pred: Iterable[float]):
    '''
    Spearman rank correlation coefficient, computed on given Pandas dataframe.
    '''
    return mean_absolute_percentage_error(y_true, y_pred) * 100


def strip_unused_amllibrary_config_sections(config: ConfigParser, techniques: list):
    for section in config.sections():
        if section in ['General', 'DataPreparation']:
            continue

        # delete config section not relevant to selected techniques
        if section not in techniques:
            del config[section]


def create_empty_folder(folder_path: str):
    '''
    Create an empty folder. If the folder already exists, it's deleted and recreated.
    Args:
        folder_path: path of the folder
    '''
    try:
        os.makedirs(folder_path)
    except OSError:
        shutil.rmtree(folder_path)
        os.makedirs(folder_path)


def chunks(lst: list, chunks_size: int):
    '''Yield successive n-sized chunks from lst.'''
    for i in range(0, len(lst), chunks_size):
        yield lst[i:i + chunks_size]


def alternative_dict_to_string(d: dict):
    ''' Avoids characters that are invalid in some OS filesystem '''
    return f'({",".join([f"{key}={value}" for key, value in d.items()])})'


def prod(it: Iterable):
    ''' Poor men's math.prod, for supporting python < 3.8. '''
    return functools.reduce(operator.mul, it, 1)


def intersection(iter1: Iterable, iter2: Iterable) -> list:
    return list(set(iter1) & set(iter2))


def to_one_hot(cat_value: int, one_hot_dim: int):
    one_hot = [0] * one_hot_dim
    one_hot[cat_value] = 1
    return one_hot


def from_seconds_to_hms(time: float):
    total_seconds = int(time)
    hours = total_seconds // 3600
    minutes = (total_seconds // 60) % 60
    seconds = total_seconds % 60

    return hours, minutes, seconds


def run_as_sequential_process(f: Callable, args: tuple = (), kwargs: dict = None):
    '''
    Utility function used to encapsulate a function into a process, waiting immediately for its conclusion (so it's not meant for parallelization!).
    It is a desperate move for encapsulating TF memory leaks and destroy them at the end of the process,
    so that really long experiments can run flawlessly.
    '''
    if kwargs is None:
        kwargs = {}

    process = Process(target=f, args=args, kwargs=kwargs)
    process.start()
    process.join()


def filter_none(*args):
    return iter(n for n in args if n is not None)
