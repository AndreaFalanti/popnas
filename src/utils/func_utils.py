# TODO: extrapolate generic helper functions used by multiple modules here

from typing import Iterable
import pandas as pd


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
