import ast
from typing import NamedTuple, Callable

from igraph import *

from models.operators.params_utils import compute_conv_params
from search_space import BlockSpecification, CellSpecification
from utils.tensor_utils import get_tensors_spatial_ratio


class TensorNode(NamedTuple):
    name: str
    shape: 'list[float, ...]'


def shape_from_str(shape_str: str):
    return ast.literal_eval(shape_str)


def merge_attribute_dicts(dict1: dict, dict2: dict):
    return {k: v + dict2[k] for k, v in dict1}


# TODO: if speed is slow, return the info about vertices and edges and then create the graph in a single shot
def build_block_dag(g: Graph, input_nodes: 'list[TensorNode]', target_shape: 'list[float, ...]',
                    block_spec: BlockSpecification, block_index: int, cell_index: int,
                    compute_layer_params: 'Callable[[str, list[float], list[float]], int]'):
    in1, op1, in2, op2 = block_spec
    node_1 = input_nodes[in1]
    node_2 = input_nodes[in2]

    # add vertices, setting convenient name identifiers
    prefix = f'c{cell_index}_b{block_index}'
    v_attributes = {
        'name': [f'{prefix}_L', f'{prefix}_R', f'{prefix}_ADD'],
        'op': [op1, op2, 'add'],
        'cell_index': [cell_index] * 3,
        'block_index': [block_index] * 3,
        'params': [compute_layer_params(op1, node_1.shape, target_shape), compute_layer_params(op2, node_2.shape, target_shape), 0]
    }
    g.add_vertices(3, v_attributes)

    # add edges from inputs to operator layers, plus from operator layers to add
    edges = [(node_1.name, f'{prefix}_L'), (node_2.name, f'{prefix}_R'),
             (f'{prefix}_L', f'{prefix}_ADD'), (f'{prefix}_R', f'{prefix}_ADD')]
    edge_attributes = {
        'tensor_shape': [str(node_1.shape), str(node_2.shape), str(target_shape), str(target_shape)]
    }
    g.add_edges(edges, edge_attributes)

    return TensorNode(f'{prefix}_ADD', target_shape)


def build_cell_dag(g: Graph, lookback_nodes: 'list[TensorNode]', target_shape: 'list[float, ...]',
                   cell_spec: CellSpecification, cell_index: int, residual_output: bool, lookback_reshape_cell_indexes: 'list[int]',
                   compute_layer_params: 'Callable[[str, list[float], list[float]], int]'):
    used_blocks = set(inp for inp in cell_spec.inputs if inp >= 0)
    unused_blocks = [b for b in range(len(cell_spec)) if b not in used_blocks]
    used_lookbacks = set([inp for inp in cell_spec.inputs if inp < 0])
    nearest_used_lookback = max(used_lookbacks)

    # automatic lookback upsample, if target shape spatial resolution is higher than the inputs
    lookback_nodes = perform_lookback_upsample_when_necessary(g, lookback_nodes, target_shape, used_lookbacks, cell_index)

    if cell_index in lookback_reshape_cell_indexes:
        lookback_nodes = perform_lookback_reshape(g, lookback_nodes, target_shape, cell_index, compute_layer_params)

    input_nodes = lookback_nodes.copy()
    block_nodes = []

    for i, block_spec in enumerate(cell_spec):
        block_output = build_block_dag(g, input_nodes, target_shape, block_spec, i, cell_index, compute_layer_params)
        block_nodes.append(block_output)
        input_nodes = block_nodes + lookback_nodes

    # concatenate multiple unused blocks or extract the one not used in the internal DAG
    if len(unused_blocks) >= 2:
        cell_out = build_cell_output_concat(g, unused_blocks, target_shape, cell_index, compute_layer_params)
    else:
        cell_out = input_nodes[unused_blocks[0]]

    if residual_output:
        cell_out = build_residual_connection(g, input_nodes[nearest_used_lookback], cell_out, cell_index, compute_layer_params)

    return cell_out


def build_cell_output_concat(g: Graph, unused_block_indexes: 'list[int]', target_shape: 'list[float, ...]', cell_index: int,
                             compute_layer_params: 'Callable[[str, list[float], list[float]], int]'):
    concat_filters = target_shape[-1] * len(unused_block_indexes)
    concat_shape = target_shape[:-1] + [concat_filters]

    v_attributes = {
        'name': [f'c{cell_index}_concat', f'c{cell_index}_out'],
        'op': ['concat', '1x1 conv'],
        'cell_index': [cell_index] * 2,
        'block_index': [-1] * 2,
        'params': [0, compute_layer_params('1x1 conv', concat_shape, target_shape)]
    }
    g.add_vertices(2, attributes=v_attributes)

    # edges from unused block adds to concat, plus from concat to output
    prefix = f'c{cell_index}_'
    edges = [(f'{prefix}b{b}_ADD', f'c{cell_index}_concat') for b in unused_block_indexes] + [(f'c{cell_index}_concat', f'c{cell_index}_out')]
    edge_attributes = {
        'tensor_shape': [str(target_shape)] * len(unused_block_indexes) + [str(concat_shape)]
    }
    g.add_edges(edges, edge_attributes)

    return TensorNode(f'c{cell_index}_out', target_shape)


def build_residual_connection(g: Graph, residual_input_node: TensorNode, residual_node: TensorNode, cell_index: int,
                              compute_layer_params: 'Callable[[str, list[float], list[float]], int]'):
    same_spatial = residual_input_node.shape[0] == residual_node.shape[0]
    same_channels = residual_input_node.shape[-1] == residual_node.shape[-1]

    if not same_channels or not same_spatial:
        residual_node = build_residual_linear_projection(g, residual_input_node, residual_node.shape, cell_index, compute_layer_params)

    residual_add_name = f'c{cell_index}_residual'
    v_attributes = {
        'name': [residual_add_name],
        'op': ['add'],
        'cell_index': [cell_index],
        'block_index': [-1],
        'params': [0]
    }
    g.add_vertices(1, attributes=v_attributes)

    edges = [(residual_node.name, residual_add_name), (residual_input_node.name, residual_add_name)]
    edge_attributes = {
        'tensor_shape': [str(residual_node.shape)] * 2
    }
    g.add_edges(edges, attributes=edge_attributes)

    return TensorNode(residual_add_name, residual_node.shape)


def build_residual_linear_projection(g: Graph, residual_input_node: TensorNode, target_shape: 'list[float, ...]', cell_index: int,
                                     compute_layer_params: 'Callable[[str, list[float], list[float]], int]'):
    same_channels = residual_input_node.shape[-1] == target_shape[-1]
    op = '2x2 maxpool' if same_channels else '1x1 conv'
    params = 0 if same_channels else compute_layer_params('1x1 conv', residual_input_node.shape, target_shape)

    res_node_name = f'c{cell_index}_residual_linear_proj'
    v_attributes = {
        'name': [res_node_name],
        'op': [op],
        'cell_index': [cell_index],
        'block_index': [-1],
        'params': [params]
    }
    g.add_vertices(1, attributes=v_attributes)

    edges = [(residual_input_node.name, res_node_name)]
    edge_attributes = {
        'tensor_shape': [str(residual_input_node.shape)]
    }
    g.add_edges(edges, attributes=edge_attributes)

    return TensorNode(res_node_name, target_shape)


def perform_lookback_reshape(g: Graph, lookback_nodes: 'list[TensorNode]', target_shape: 'list[float, ...]', cell_index: int,
                             compute_layer_params: 'Callable[[str, list[float], list[float]], int]'):
    lb2, lb1 = lookback_nodes
    # TODO: target should be always equal to lb1.shape when available, so probably could avoid the ternary operator
    new_shape = target_shape if lb1 is None else lb1.shape

    reshape_name = f'c{cell_index}_lb2_reshape'
    v_attributes = {
        'name': [reshape_name],
        'op': ['1x1 conv'],
        'cell_index': [cell_index],
        'block_index': [-1],
        'params': [compute_layer_params('1x1 conv', lb2.shape, new_shape)]
    }
    g.add_vertices(1, attributes=v_attributes)

    edges = [(lb2.name, reshape_name)]
    edge_attributes = {
        'tensor_shape': [str(lb2.shape)]
    }
    g.add_edges(edges, attributes=edge_attributes)

    # replace lb2 with the reshaped one, having now the same shape of lb1
    return [TensorNode(reshape_name, new_shape), lb1]


def perform_lookback_upsample_when_necessary(g: Graph, lookback_nodes: 'list[TensorNode]', target_shape: 'list[float, ...]',
                                             used_lookbacks: 'set[int]', cell_index: int):
    new_lookbacks = []
    for i, lb in enumerate(lookback_nodes):
        lb_pos = len(lookback_nodes) - i
        # avoid upsampling unused lookbacks (for example, -2 when only -1 is used)
        if lb is None or -lb_pos not in used_lookbacks:
            new_lookbacks.append(lb)
        else:
            spatial_ratio = round(get_tensors_spatial_ratio(target_shape, lb.shape))
            if spatial_ratio > 1:
                upsample_name = f'c{cell_index}_lb{lb_pos}_transpose_conv_upsample'
                # TODO: adapt kernel for time series (1D)
                v_attributes = {
                    'name': [upsample_name],
                    'op': [f'{spatial_ratio}x{spatial_ratio} tconv'],
                    'cell_index': [cell_index],
                    'block_index': [-1],
                    # do not use tconv allocator formula, since it stacks two convolutions to keep the same size, while here it is used properly!
                    'params': [compute_conv_params((spatial_ratio, spatial_ratio), int(lb.shape[-1]), int(target_shape[-1]))]
                }
                g.add_vertices(1, attributes=v_attributes)

                edges = [(lb.name, upsample_name)]
                edge_attributes = {
                    'tensor_shape': [str(lb.shape)]
                }
                g.add_edges(edges, attributes=edge_attributes)

                new_lookbacks.append(TensorNode(upsample_name, target_shape))
            else:
                new_lookbacks.append(lb)

    return new_lookbacks
