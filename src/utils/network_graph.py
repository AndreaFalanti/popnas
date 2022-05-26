# NetworkGraph involves some logic duplication with the ModelGenerator, since it must support
# the same exact structure. This means that any change to the ModelGenerator must also be reflected
# in this module.

import re
from collections import namedtuple
from typing import Union

from igraph import *

from utils.func_utils import list_flatten, chunks

TensorShape = namedtuple('TensorShape', 'h, w, c')

op_regex_dict = {'conv': re.compile(r'(\d+)x(\d+) conv'),
                 'dconv': re.compile(r'(\d+)x(\d+) dconv'),
                 'tconv': re.compile(r'(\d+)x(\d+) tconv'),
                 'stack_conv': re.compile(r'(\d+)x(\d+)-(\d+)x(\d+) conv'),
                 'pool': re.compile(r'(\d+)x(\d+) (max|avg)pool')}


def compute_batch_norm_params(filters: int):
    return 4 * filters


def compute_conv_params(kernel_h: Union[int, str], kernel_w: Union[int, str], input_shape: TensorShape, output_shape: TensorShape):
    ''' Actually is convolution + batch normalization, since a convolution is always followed by bn in the model. '''
    # +1 is bias term
    return (int(kernel_h) * int(kernel_w) * input_shape.c + 1) * output_shape.c + compute_batch_norm_params(output_shape.c)


def compute_dconv_params(kernel_h: Union[int, str], kernel_w: Union[int, str], input_shape: TensorShape, output_shape: TensorShape):
    ''' Depthwise separable convolution + batch norm. '''
    # bias term is used only in pointwise for unknown reasons, also has no batch normalization so it is computed without "compute_conv_params"
    return (int(kernel_h) * int(kernel_w) * input_shape.c) + \
           compute_conv_params(1, 1, input_shape, output_shape)


def compute_op_params(op: str, input_shape: TensorShape, output_shape: TensorShape):
    preserve_shape = input_shape == output_shape
    # this operators can't have parameters in any case
    no_params_operators = ['add', 'concat', 'input', 'gap']

    if op in no_params_operators:
        return 0

    # if identity need to change shape, then it becomes a pointwise convolution
    if op == 'identity':
        return 0 if preserve_shape else compute_conv_params(1, 1, input_shape, output_shape)

    # single convolution case
    match = op_regex_dict['conv'].match(op)  # type: re.Match
    if match:
        return compute_conv_params(match.group(1), match.group(2), input_shape, output_shape)

    # pooling case, if needs to change shape, then it is followed by a pointwise convolution
    match = op_regex_dict['pool'].match(op)  # type: re.Match
    if match:
        return 0 if preserve_shape else compute_conv_params(1, 1, input_shape, output_shape)

    # stacked convolution case, both use batch normalization and only the first one modifies the shape
    match = op_regex_dict['stack_conv'].match(op)  # type: re.Match
    if match:
        return compute_conv_params(match.group(1), match.group(2), input_shape, output_shape) + \
               compute_conv_params(match.group(3), match.group(4), output_shape, output_shape)

    # depthwise separable convolution case
    match = op_regex_dict['dconv'].match(op)  # type: re.Match
    if match:
        return compute_dconv_params(match.group(1), match.group(2), input_shape, output_shape)

    match = op_regex_dict['tconv'].match(op)  # type: re.Match
    if match:
        # TODO
        raise NotImplementedError('TODO, tconv is not used in recent experiments even if supported')

    raise AttributeError(f'Unsupported operator "{op}"')


def compute_target_shapes(input_shape: TensorShape, cells_count: int, filters: int,
                          reduction_cell_indices: 'list[int]', shape_tx: 'tuple[float, float, float]') -> 'list[TensorShape]':
    output_shapes = []

    # replace last dimension with the filters
    input_shape = TensorShape(input_shape.h, input_shape.w, filters)
    output_shape = input_shape

    for cell_index in range(cells_count):
        if cell_index in reduction_cell_indices:
            output_shape = TensorShape(*(math.floor(a * b) for a, b in zip(output_shape, shape_tx)))

        output_shapes.append(output_shape)

    return output_shapes


# TODO: inspired by feature_utils.compute_features_from_dag, maybe it is better to avoid duplication of first part of code
def create_cell_graph(cell_spec: list, cell_index: int, output_shape: TensorShape):
    blocks = len(cell_spec)
    flat_cell_spec = list_flatten(cell_spec)
    inputs = flat_cell_spec[::2]
    operators = flat_cell_spec[1::2]

    # internal edges of each block, without connections with the lookbacks (done in merge_graphs)
    basic_edges = list_flatten([[(i * 3, i * 3 + 2), (i * 3 + 1, i * 3 + 2)] for i in range(blocks)])
    block_dep_edges = [(2 + 3 * inp, i + (i // 2)) for i, inp in enumerate(inputs) if inp >= 0]

    # for each block the 2 operations plus the add (3 vertices)
    v_count = 3 * blocks
    e_count = len(basic_edges + block_dep_edges)
    edge_attributes = {
        'tensor_h': [output_shape.h] * e_count,
        'tensor_w': [output_shape.w] * e_count,
        'tensor_c': [output_shape.c] * e_count
    }

    #  set vertex names to make them recognizable and also easier to select, plus other attributes
    v_attributes = {
        'name': list_flatten([(f'c{cell_index}_b{b}_L', f'c{cell_index}_b{b}_R', f'c{cell_index}_b{b}_ADD') for b in range(blocks)]),
        'op': list_flatten([(op1, op2, 'add') for op1, op2 in chunks(operators, 2)]),
        'cell_index': [cell_index] * v_count,
        'block_index': list_flatten([[b] * 3 for b in range(blocks)])
    }

    g = Graph(n=v_count, edges=basic_edges + block_dep_edges, directed=True, edge_attrs=edge_attributes, vertex_attrs=v_attributes)

    # utility field for connecting graphs (need to connect lookbacks to main graph)
    g.vs['connect_lookback'] = list_flatten([(in1, in2, 0) for in1, in2 in chunks(inputs, 2)])

    add_layer_vertices = g.vs.select([2 + 3 * i for i in range(blocks)])  # type: VertexSeq
    unused_adds = add_layer_vertices.select(_outdegree_eq=0)  # type: VertexSeq

    # if multiple block outputs are unused, add concat and pointwise convolution to graph
    # connect_lookback field here is fictitious, to avoid null and also to avoid recompute params later
    if len(unused_adds) >= 2:
        concat_shape = TensorShape(output_shape.h, output_shape.w, len(unused_adds) * output_shape.c)
        v_attributes = {
            'name': [f'concat_c{cell_index}', f'out_c{cell_index}'],
            'op': ['concat', '1x1 conv'],
            'cell_index': [cell_index] * 2,
            'connect_lookback': [-100, -100],
            'params': [0, compute_conv_params(1, 1, concat_shape, output_shape)]
        }

        g.add_vertices(2, attributes=v_attributes)
        # concat will have as index 'v_count', since it is the first added vertex and they are 0-indexed
        adds_to_concat_edges = [(v.index, v_count) for v in unused_adds]

        # add also edge between concat and pointwise, plus generate the attributes (concat will have as number of channels output_c * #_adds)
        edge_attributes = {
            'tensor_h': [output_shape.h] * (len(adds_to_concat_edges) + 1),
            'tensor_w': [output_shape.w] * (len(adds_to_concat_edges) + 1),
            'tensor_c': [output_shape.c] * len(adds_to_concat_edges) + [output_shape.c * len(adds_to_concat_edges)]
        }

        g.add_edges(adds_to_concat_edges + [(v_count, v_count + 1)], attributes=edge_attributes)
    else:
        unused_adds['name'] = [f'out_c{cell_index}']

    return g


def merge_graphs(main_g: Graph, cell_g: Graph, lookback_indexes: 'list[Union[int, str]]',
                 lookback_shapes: 'list[TensorShape]', output_shape: TensorShape):
    # if shapes differ (and are not related to original input), then -2 lookback must be reshaped to the same dimension of -1 lookback
    # only height is checked since only one dimension is necessary to diverge to trigger the reshaping
    if not lookback_shapes[-1].h == lookback_shapes[-2].h and not lookback_indexes[-2] == 'input':
        layer_name = f'rs_{lookback_indexes[-2]}'

        main_g.add_vertex(layer_name, op='1x1 conv', params=compute_conv_params(1, 1, lookback_shapes[-2], lookback_shapes[-1]))
        main_g.add_edge(lookback_indexes[-2], layer_name,
                        tensor_h=lookback_shapes[-2].h, tensor_w=lookback_shapes[-2].w, tensor_c=lookback_shapes[-2].c)

        lookback_indexes[-2] = layer_name
        lookback_shapes[-2] = lookback_shapes[-1]

    lb1_vs = cell_g.vs.select(connect_lookback=-1)
    lb2_vs = cell_g.vs.select(connect_lookback=-2)
    internal_vs = cell_g.vs.select(connect_lookback_ge=0)

    # compute parameters of all operations of the cell
    lb1_vs['params'] = [compute_op_params(op, lookback_shapes[-1], output_shape) for op in lb1_vs['op']]
    lb2_vs['params'] = [compute_op_params(op, lookback_shapes[-2], output_shape) for op in lb2_vs['op']]
    internal_vs['params'] = [compute_op_params(op, output_shape, output_shape) for op in internal_vs['op']]

    g = main_g.disjoint_union(cell_g)  # type: Graph

    # join the graphs through edges involving lookbacks (values >= 0 are not relevant)
    lb1_edges = [(lookback_indexes[-1], v.index) for v in lb1_vs]
    lb2_edges = [(lookback_indexes[-2], v.index) for v in lb2_vs]

    edge_attributes = {
        'tensor_h': [lookback_shapes[-1].h] * len(lb1_edges) + [lookback_shapes[-2].h] * len(lb2_edges),
        'tensor_w': [lookback_shapes[-1].w] * len(lb1_edges) + [lookback_shapes[-2].w] * len(lb2_edges),
        'tensor_c': [lookback_shapes[-1].c] * len(lb1_edges) + [lookback_shapes[-2].c] * len(lb2_edges)
    }

    g.add_edges(lb1_edges + lb2_edges, attributes=edge_attributes)
    # remove utility field otherwise it will be used in next merge, causing wrong edge connections
    del g.vs['connect_lookback']

    return g


class NetworkGraph:
    '''
    Generate and store a graph representing the whole neural network structure, without the learning functionalities
    or the utilities to allocate the layers in memory, for building the actual network use Keras instead (ModelGenerator).

    This class is intended to produce a graph with a 1000x speedup compared to generating it in keras, providing
    a standard graph structure which can be analyzed with operational research techniques.

    It is also useful to estimate very fast the amount of memory required by the neural network.
    '''
    def __init__(self, cell_spec: list, input_shape: 'tuple[int, int, int]', filters: int,
                 num_classes: int, motifs: int, normals_per_motif: int) -> None:
        super().__init__()

        flat_inputs = list_flatten(cell_spec)[::2]
        cells_count = motifs * (normals_per_motif + 1) - 1

        used_lookbacks = set(filter(lambda el: el < 0, flat_inputs))
        # use_skips = any(x < -1 for x in used_lookbacks)
        used_cell_indices = list(range(cells_count - 1, -1, max(used_lookbacks, default=cells_count)))

        reduction_cell_indices = list(range(normals_per_motif, cells_count, normals_per_motif + 1))
        reduction_shape_transform = (0.5, 0.5, 2)

        input_shape = TensorShape(*input_shape)
        target_shapes = compute_target_shapes(input_shape, cells_count, filters, reduction_cell_indices, reduction_shape_transform)

        # initialize the graph with only the input node
        g = Graph(n=1, directed=True)
        g.vs['op'] = ['input']
        g.vs['params'] = [0]
        g.vs['cell_index'] = [-1]
        g.vs['block_index'] = [-1]
        g.vs['name'] = ['input']

        # use integer indices initially, then it's easier to use the names (select support index or name (string vertex attribute))
        lookback_vertex_indices = [0, 0]
        lookback_shapes = [input_shape, input_shape]

        # add input shape normalization if using skips
        # if use_skips:
        #     # TODO: params
        #     g.add_vertex('rs_input', type='1x1 conv', params=123, cell_index=-1, block_index=-1)
        #     g.add_edges([(0, 1)])
        #     lookback_vertex_indices = [1, 0]

        # iterate on cells, using reduction bool list since we need the value later
        for cell_index, output_shape in enumerate(target_shapes):
            if cell_index in used_cell_indices:
                cell_g = create_cell_graph(cell_spec, cell_index, output_shape)
                g = merge_graphs(g, cell_g, lookback_vertex_indices, lookback_shapes, output_shape)

            # move -1 lookback to -2 position, than add new cell output as -1 lookback
            lookback_vertex_indices[-2] = lookback_vertex_indices[-1]
            lookback_vertex_indices[-1] = f'out_c{cell_index}'

            # same for shapes
            lookback_shapes[-2] = lookback_shapes[-1]
            lookback_shapes[-1] = output_shape

        # add final GAP and Softmax
        v_attributes = {
            'name': ['GAP', 'Softmax'],
            'op': ['gap', 'softmax'],
            'params': [0, num_classes * (output_shape.c + 1)]
        }
        g.add_vertices(2, attributes=v_attributes)

        # connect last cell to GAP and GAP to Softmax
        edge_attributes = {
            'tensor_h': [1, 1],
            'tensor_w': [1, 1],
            'tensor_c': [output_shape.c, num_classes]
        }
        g.add_edges([(lookback_vertex_indices[-1], 'GAP'), ('GAP', 'Softmax')], attributes=edge_attributes)

        # print(g)
        self.g = g

    def get_total_params(self):
        # print(self.g.vs['params'])
        return sum(self.g.vs['params'])
