# TODO: extrapolate generic helper functions used by multiple modules here

from typing import Iterable


def to_int_tuple(str_tuple: 'tuple[str, ...]'):
    '''
    Cast each str element of a tuple to int and return it.
    '''
    return tuple(map(int, str_tuple))


def list_flatten(nested_l: 'list[Iterable]'):
    return [el for iter in nested_l for el in iter]
