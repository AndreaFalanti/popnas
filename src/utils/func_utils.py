# TODO: extrapolate generic helper functions used by multiple modules here

def to_int_tuple(str_tuple: 'tuple[str, ...]'):
    '''
    Cast each element of a tuple to int and return it.
    '''
    return tuple(map(int, str_tuple))
