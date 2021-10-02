# TODO: extrapolate generic helper functions used by multiple modules here
import re
from typing import Iterable
import pandas as pd

from sklearn.metrics import mean_absolute_percentage_error


def to_int_tuple(str_tuple: 'tuple[str, ...]'):
    '''
    Cast each str element of a tuple to int and return it.
    '''
    return tuple(map(int, str_tuple))


def list_flatten(nested_l: 'list[Iterable]'):
    return [el for iterable_el in nested_l for el in iterable_el]


def to_list_of_tuples(sequence, chunk_size):
    return list(zip(*[iter(sequence)] * chunk_size))


def clamp(n: float, lower_bound: float, upper_bound: float):
    return max(lower_bound, min(n, upper_bound))


def compute_spearman_rank_correlation_coefficient(df: pd.DataFrame, x_col: str, y_col: str):
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
    return [list(tuple(map(lambda str_tuple: tuple(str_tuple.split(', ')), tuple_str_list)))
            for tuple_str_list in list_of_tuple_str_lists]
