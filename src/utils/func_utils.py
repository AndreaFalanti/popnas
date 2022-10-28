import functools
import json
import operator
import os
import re
import shutil
from configparser import ConfigParser
from typing import Iterable

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


def compute_spearman_rank_correlation_coefficient(y_true: 'list[float]', y_pred: 'list[float]'):
    '''
    Spearman rank correlation coefficient, given two lists.
    '''
    df = pd.DataFrame.from_dict({'true': y_true, 'est': y_pred})
    return df['true'].corr(df['est'], method='spearman')


def compute_spearman_rank_correlation_coefficient_from_df(df: pd.DataFrame, x_col: str, y_col: str):
    '''
    Spearman rank correlation coefficient, computed on given Pandas dataframe.
    '''
    return df[x_col].corr(df[y_col], method='spearman')


def compute_mape(y_true: 'list[float]', y_pred: 'list[float]'):
    '''
    Spearman rank correlation coefficient, computed on given Pandas dataframe.
    '''
    return mean_absolute_percentage_error(y_true, y_pred) * 100


def parse_cell_structures(cell_structures: Iterable):
    '''
    Function used to parse in an actual python structure the csv field storing the non-encoded cell structure, which is saved in form:
    "[(in1,op1,in2,op2);(...);...]"
    Args:
        cell_structures:

    Returns:
        (list[list]): list of lists of tuples, a list of tuples for each cell parsed

    '''
    # remove set of chars { []()'" } for easier parsing
    cell_structures = list(map(lambda cell_str: re.sub(r'[\[\]\'\"()]', '', cell_str), cell_structures))
    # parse cell structure into multiple strings, each one representing a tuple
    list_of_tuple_str_lists = list(map(lambda cs: cs.split(';'), cell_structures))

    # parse tuple structure (trim round brackets and split by ,)
    str_cell_specs = [list(tuple(map(lambda str_tuple: tuple(str_tuple.split(', ')), tuple_str_list)))
                      for tuple_str_list in list_of_tuple_str_lists]

    # fix cell structure having inputs as str type instead of int
    adjusted_cells = []
    for cell in str_cell_specs:
        # initial thrust case, empty cell
        if cell == [('',)]:
            adjusted_cells.append([])
        else:
            adjusted_cells.append([(int(in1), op1, int(in2), op2) for in1, op1, in2, op2 in cell])

    return adjusted_cells


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


def get_valid_inputs_for_block_size(input_values: list, current_blocks: int, max_blocks: int):
    ''' Remove inputs that can't be assigned in current blocks step (example: 1,2,3 for current_blocks=2) '''
    inputs_to_prune_count = current_blocks - max_blocks
    return input_values if inputs_to_prune_count >= 0 else input_values[:inputs_to_prune_count]


def alternative_dict_to_string(d: dict):
    ''' Avoids characters that are invalid in some OS filesystem '''
    return f'({",".join([f"{key}={value}" for key, value in d.items()])})'


def instantiate_search_space_from_logs(log_folder_path: str):
    '''
    Instantiate a SearchSpace instance, using the settings specified in the run configuration saved in a log folder.

    Args:
        log_folder_path: path to log folder

    Returns:
        (SearchSpace): the search space instance
    '''
    # import must be done here, to avoid circular dependency
    from encoder import SearchSpace

    run_config_path = os.path.join(log_folder_path, 'restore', 'run.json')
    with open(run_config_path, 'r') as f:
        run_config = json.load(f)

    ss_config = run_config['search_space']
    arc_config = run_config['architecture_parameters']
    max_cells = arc_config['motifs'] * (arc_config['normal_cells_per_motif'] + 1) - 1

    return SearchSpace(ss_config, max_cells)


def cell_spec_to_str(cell_spec: list):
    return f"[{';'.join(map(str, cell_spec))}]"


def prod(it: Iterable):
    ''' Poor men's math.prod, for supporting python < 3.8. '''
    return functools.reduce(operator.mul, it, 1)


def intersection(iter1: Iterable, iter2: Iterable) -> list:
    return list(set(iter1) & set(iter2))


def to_one_hot(cat_value: int, one_hot_dim: int):
    one_hot = [0] * one_hot_dim
    one_hot[cat_value] = 1
    return one_hot
