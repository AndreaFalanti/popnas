# Module that contains helper functions for producing some regressor features
import csv
import os.path
from typing import Callable

import igraph

from search_space import SearchSpace
from search_space_units import CellSpecification
from utils.func_utils import list_flatten, to_list_of_tuples


# region FEATURE_NAMES_AND_FILES_INITIALIZATION
def build_time_feature_names():
    headers = ['time'] + ['blocks', 'cells', 'op_score', 'multiple_lookbacks', 'first_dag_level_op_score_fraction', 'block_dependencies',
                          'dag_depth', 'concat_inputs', 'heaviest_dag_path_op_score_fraction'] + ['exploration']
    header_types = ['Label'] + ['Num'] * 9 + ['Auxiliary']

    return headers, header_types


def build_score_feature_names(max_blocks: int, max_lookback: int):
    op_headers, op_headers_types = [], []
    block_incidence_headers, block_incidence_headers_types = [], []
    lookback_incidence_headers, lookback_incidence_headers_types = [], []

    for i in range(max_blocks):
        op_headers.extend([f'op{i * 2}', f'op{i * 2 + 1}'])

        for j in range(i):
            block_incidence_headers.append(f'b{j}b{i}')

        for lb in range(max_lookback):
            lookback_incidence_headers.append(f'lb{lb + 1}b{i}')

    lookback_usage_headers = [f'lb{i + 1}' for i in range(max_lookback)]

    op_headers_types = ['Categ'] * len(op_headers)
    block_incidence_headers_types = ['Num'] * len(block_incidence_headers)
    lookback_incidence_headers_types = ['Num'] * len(lookback_incidence_headers)
    lookback_usage_headers_types = ['Num'] * len(lookback_usage_headers)

    headers = ['score', 'blocks', 'cells'] + op_headers + lookback_usage_headers + \
              lookback_incidence_headers + block_incidence_headers + ['exploration', 'data_augmented']
    header_types = ['Label', 'Num', 'Num'] + op_headers_types + lookback_usage_headers_types + \
                   lookback_incidence_headers_types + block_incidence_headers_types + ['Auxiliary', 'Auxiliary']

    return headers, header_types


def initialize_features_csv_files(time_headers: list, time_feature_types: list, acc_headers: list, acc_feature_types: list, csv_folder_path: str):
    with open(os.path.join(csv_folder_path, f'column_desc_time.csv'), mode='w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(enumerate(time_feature_types))

    with open(os.path.join(csv_folder_path, f'column_desc_acc.csv'), mode='w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(enumerate(acc_feature_types))

    with open(os.path.join(csv_folder_path, 'training_time.csv'), mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(time_headers)

    with open(os.path.join(csv_folder_path, 'training_time_inference.csv'), mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(time_headers)

    with open(os.path.join(csv_folder_path, 'training_score.csv'), mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(acc_headers)
# endregion


# region TIME_FEATURES_GEN
def generate_dynamic_reindex_function(op_timers: 'dict[str, float]', initial_thrust_time: float):
    '''
    Closure for generating a function to easily apply dynamic reindex where necessary.

    Args:
        initial_thrust_time: training time of CNN generated from empty cell specification
        op_timers: dict with op as key and time as value

    Returns:
        Callable[[str], float]: dynamic reindex function
    '''
    for key, val in op_timers.items():
        op_timers[key] = abs(val - initial_thrust_time)

    t_max = max(op_timers.values())

    def apply_dynamic_reindex(op_value: str):
        return op_timers[op_value] / t_max

    return apply_dynamic_reindex


def compute_features_from_dag(inputs: 'list[int]', op_features: 'list[tuple[int, int]]', total_op_score: float):
    blocks = len(op_features)
    flat_ops = list_flatten(op_features)

    # internal edges of each block, without connections with the inputs
    basic_edges = list_flatten([[(i*3, i*3 + 2), (i*3 + 1, i*3 + 2)] for i in range(blocks)])
    block_dep_edges = [(2 + 3 * inp, i + (i // 2)) for i, inp in enumerate(inputs) if inp >= 0]

    # for each block the 2 operations plus the add (3 vertices)
    g = igraph.Graph(n=3 * blocks, edges=basic_edges + block_dep_edges, directed=True)

    # add op weight to each graph vertex (0 for add)
    g.vs['op_time'] = list_flatten([(op1, op2, 0) for op1, op2 in op_features])

    # add nodes should not be considered
    dag_depth = 1 + g.diameter() // 2

    add_vertices = g.vs.select([2 + 3 * i for i in range(blocks)])  # type: igraph.VertexSeq
    adds_in_cell_outputs = add_vertices.select(_outdegree_eq=0)
    concat_inputs_len = len(adds_in_cell_outputs) if len(adds_in_cell_outputs) > 1 else 0   # if 1, there is no concat

    # get heaviest path
    if dag_depth == 1:
        heaviest_path_op_score_fraction = max(flat_ops) / total_op_score
    else:
        heaviest_path_op_score_fraction = 0
        # get all vertexes involved in paths to reach one of the ops of the adds involved in final concatenation
        for av in adds_in_cell_outputs:
            vertexes_that_reach_op1 = g.subcomponent(av.index - 2, mode='in')
            vertexes_that_reach_op2 = g.subcomponent(av.index - 1, mode='in')

            op1_path_op_score = sum(g.vs.select(vertexes_that_reach_op1)['op_time'])
            op2_path_op_score = sum(g.vs.select(vertexes_that_reach_op2)['op_time'])
            heaviest_path_op_score_fraction = max(heaviest_path_op_score_fraction, op1_path_op_score, op2_path_op_score)

        heaviest_path_op_score_fraction = heaviest_path_op_score_fraction / total_op_score

    return [dag_depth, concat_inputs_len, heaviest_path_op_score_fraction]


def generate_time_features(cell_spec: CellSpecification, search_space: SearchSpace,
                           reindex_type: str, compute_real_cell_depth: Callable[[CellSpecification], int]):
    inputs = cell_spec.inputs
    blocks = len(cell_spec)
    total_cells = compute_real_cell_depth(cell_spec)

    op_time_features_flat = search_space.encode_cell_spec(cell_spec, op_enc_name=reindex_type).operators
    op_time_features = to_list_of_tuples(op_time_features_flat, 2)

    total_op_score = sum(op_time_features_flat)
    use_different_lookbacks = 1 if len(set([inp for inp in inputs if inp < 0])) > 1 else 0
    first_level_op_score_fraction = sum([op for inp, op in zip(inputs, op_time_features_flat) if inp < 0]) / total_op_score
    block_dependencies = len([inp for inp in inputs if inp >= 0])   # with also duplications

    return [blocks, total_cells, total_op_score, use_different_lookbacks, first_level_op_score_fraction, block_dependencies]\
           + compute_features_from_dag(inputs, op_time_features, total_op_score)
# endregion


# region ACC_FEATURES_GEN
def compute_blocks_incidence_matrix(cell_spec: CellSpecification, max_b: int):
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


def compute_lookback_usage_features(cell_spec: CellSpecification, max_lookback: int):
    '''
    Produce a list of int (0 or 1), that states if a lookback depth is used in the cell specification.
    List starts from lowest depth usage (-1).

    Args:
        cell_spec: plain cell specification
        max_lookback: max lookback distance, in positive value

    Returns:
        (list[int]): list of boolean (0 or 1) features about lookback usage
    '''
    # remove inputs = None
    cell_inputs = [inp for inp in cell_spec.inputs if inp is not None]

    return [(1 if -i - 1 in cell_inputs else 0) for i in range(max_lookback)]


def compute_blocks_lookback_incidence_matrix(cell_spec: CellSpecification, max_b: int, max_lookback: int):
    '''
    Compute the incidence matrix of lookback inputs in blocks, that can be used as a list of features in ML algorithms.

    Args:
        cell_spec: plain cell specification
        max_b: maximum number of blocks in a cell
        max_lookback: max lookback distance, in positive value

    Returns:
        (list[int]): list of incidence boolean features as integers (0 or 1)
    '''
    incidence_features = [0] * (max_b * max_lookback)

    for i, (in1, _, in2, _) in enumerate(cell_spec):
        for lb in range(max_lookback):
            incidence_features[i * max_lookback + lb] = 1 if -lb - 1 in [in1, in2] else 0

    return incidence_features


def generate_acc_features(cell_spec: CellSpecification, search_space: SearchSpace, compute_real_cell_depth: Callable[[CellSpecification], int]):
    blocks = len(cell_spec)

    op_features = search_space.encode_cell_spec(cell_spec).operators
    max_blocks = search_space.B
    max_lookback_depth = abs(search_space.input_lookback_depth)

    total_cells = compute_real_cell_depth(cell_spec.prune_fictitious_blocks())
    lookback_usage_features = compute_lookback_usage_features(cell_spec, max_lookback_depth)
    lookback_incidence_features = compute_blocks_lookback_incidence_matrix(cell_spec, max_blocks, max_lookback_depth)
    block_incidence_features = compute_blocks_incidence_matrix(cell_spec, max_blocks)

    return [blocks, total_cells] + op_features + lookback_usage_features + lookback_incidence_features + block_incidence_features
# endregion
