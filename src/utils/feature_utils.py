# Module that contains helper functions for producing some regressor features
import math

from utils.func_utils import list_flatten


def compute_real_cnn_cell_stack_depth(cell_spec: list, max_cells: int):
    '''
    Compute the real amount of cells stacked in a CNN, based on the cell specification and the actual cell stack target.
    Usually the number of cells stacked in a CNN is the target imposed, but if the inputs use only lookback input values < -1,
    then some of them are actually skipped, leading to a CNN with less cells than the imposed number.

    Args:
        cell_spec: plain cell specification
        max_cells: cells to be stacked in a CNN, based on the run configuration

    Returns:
        (int): number of cells actually stacked in the CNN for given cell specification
    '''
    lookback_inputs = [inp for inp in list_flatten(cell_spec)[::2] if inp is not None and inp < 0]
    nearest_lookback_abs = abs(max(lookback_inputs))

    return math.ceil(max_cells / nearest_lookback_abs)


def compute_blocks_incidence_matrix(cell_spec: list, max_b: int):
    '''
    Compute the useful part of blocks incidence matrix (lower triangular), that can be used as a list of features in ML algorithms.

    Args:
        cell_spec: plain cell specification
        max_b: maximum amount of blocks in a cell

    Returns:
        (list[int]): list of incidence boolean features as integers (0 or 1)
    '''
    # values of the lower triangular matrix, initialized to 0
    features_count = sum(range(1, max_b))
    incidence_features = [0] * features_count

    for block_index, (in1, _, in2, _) in enumerate(cell_spec):
        # first block can't contain inputs from other blocks
        if block_index == 0:
            continue

        # draw the matrix on paper, you will see the offset follow this rule
        index_offset = sum(range(0, block_index))
        # set the correct feature to 1 if there is a block dependency. No problems if in1 = in2, 1 will be just written 2 times in same position.
        if in1 is not None and in1 >= 0:
            incidence_features[index_offset + in1] = 1
        if in2 is not None and in2 >= 0:
            incidence_features[index_offset + in2] = 1

    return incidence_features


def compute_lookback_usage_features(cell_spec: list, max_lookback: int):
    '''
    Produce a list of int (0 or 1), that states if a lookback depth is used in the cell specification.
    List starts from lowest depth usage (-1).

    Args:
        cell_spec: plain cell specification
        max_lookback: max lookback distance, in positive value

    Returns:
        (list[int]): list of boolean (0 or 1) features about lookback usage
    '''
    cell_inputs = list_flatten(cell_spec)[::2]
    # remove inputs = None
    cell_inputs = [inp for inp in cell_inputs if inp is not None]

    return [(1 if -i - 1 in cell_inputs else 0) for i in range(max_lookback)]


def compute_blocks_lookback_incidence_matrix(cell_spec: list, max_b: int, max_lookback: int):
    '''
    Compute the incidence matrix of lookback inputs in blocks, that can be used as a list of features in ML algorithms.

    Args:
        cell_spec: plain cell specification
        max_b: maximum amount of blocks in a cell
        max_lookback: max lookback distance, in positive value

    Returns:
        (list[int]): list of incidence boolean features as integers (0 or 1)
    '''
    incidence_features = [0] * (max_b * max_lookback)

    for i, (in1, _, in2, _) in enumerate(cell_spec):
        for lb in range(max_lookback):
            incidence_features[i * max_lookback + lb] = 1 if -lb - 1 in [in1, in2] else 0

    return incidence_features
