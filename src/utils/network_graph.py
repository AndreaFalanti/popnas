'''
NetworkGraph involves some logic duplication with the ModelGenerator, since it must support the same exact structure.
This means that any change to the ModelGenerator must also be reflected in this module.
TODO WORKAROUND: to support 1D operators, tensor shapes have been kept in 3D but setting 'w' to 1, it should work fine
 but a refactor could make the code more clear.
'''
import re
from collections import namedtuple
from typing import Union

from igraph import *

from utils.func_utils import list_flatten, chunks, to_int_tuple, prod

# in case the operators are 1D, if w is set to 1 it still works correctly.
# the only important thing for computing correctly the params is that c is always mapped to the number of filters.
# TODO: could be expanded to (h, w, d, c) to support volumes (3D operators)
TensorShape = namedtuple('TensorShape', 'h, w, c')

reduction_shape_transform = (0.5, 0.5, 2)


def compute_batch_norm_params(filters: int):
    return 4 * filters


def compute_layer_norm_params(filters: int):
    # beta and gamma for each channel (since layer norm is applied on axis -1)
    return 2 * filters


def compute_lstm_params(filters_in: int, filters_out: int):
    # lstm params + batch normalization that follows the rnn
    return 4 * (filters_out * filters_in + filters_out ** 2 + filters_out) + compute_batch_norm_params(filters_out)


def compute_gru_params(filters_in: int, filters_out: int):
    # lstm params + batch normalization that follows the rnn
    return 3 * (filters_out * filters_in + filters_out ** 2 + 2 * filters_out) + compute_batch_norm_params(filters_out)


def compute_conv_params(kernel: Union[str, 'tuple[Union[str, int], ...]'], filters_in: int, filters_out: int, bias: bool = True, bn: bool = True):
    '''
    Formula for computing the parameters of many convolutional operators.

    Args:
        kernel: kernel size
        filters_in: input filters
        filters_out: output filters
        bias: use bias or not
        bn: followed by BatchNormalization or not
    '''
    # split kernel in multiple digits, if in str format
    if isinstance(kernel, str):
        kernel = kernel.split('x')

    kernel = to_int_tuple(kernel)  # cast to int in case are elements are str
    b = 1 if bias else 0
    bn_params = compute_batch_norm_params(filters_out) if bn else 0

    return (prod(kernel) * filters_in + b) * filters_out + bn_params


def compute_dconv_params(kernel: Union[str, 'tuple[Union[str, int], ...]'], filters_in: int, filters_out: int, bias: bool = True, bn: bool = True):
    ''' Depthwise separable convolution + batch norm. '''
    # bias term is used only in pointwise for unknown reasons, also it has no batch normalization,
    # so it is computed separately without "compute_conv_params" function.

    # split kernel in multiple digits, if in str format
    if isinstance(kernel, str):
        kernel = kernel.split('x')

    kernel = to_int_tuple(kernel)  # cast to int in case are elements are str
    return (prod(kernel) * filters_in) + \
           compute_conv_params((1, 1), filters_in, filters_out, bias=bias, bn=bn)


def compute_cvt_params(kernel: 'tuple[Union[str, int], ...]', heads: int, blocks: int, filters_in: int, filters_out: int, use_mlp: bool):
    ''' Actually is convolution + batch normalization, since a convolution is always followed by bn in the model. '''
    # +1 is bias term
    kernel = to_int_tuple(kernel)  # cast to int in case are elements are str
    dim_head = filters_out  # forced by actual implementation (see op_instantiator)

    embed_conv_params = compute_conv_params(kernel, filters_in, filters_out, bn=False)
    layer_norm_params = compute_layer_norm_params(filters_out)

    # NOTE: dconv here actually use bn, but is in the middle instead of the end, so they use filters_in as num of channels!
    # done separately to avoid error in computation
    q_conv_params = compute_dconv_params((3, 3), filters_out, dim_head * heads, bias=False, bn=False)
    kv_conv_params = compute_dconv_params((3, 3), filters_out, 2 * dim_head * heads, bias=False, bn=False)
    bn_params = compute_batch_norm_params(filters_out) * 2
    conv_out = compute_conv_params((1, 1), dim_head * heads, filters_out, bn=False)

    mlp_mult = 2
    mlp_params = (2 * layer_norm_params +
                  compute_conv_params((1, 1), filters_out, filters_out * mlp_mult, bn=False) +
                  compute_conv_params((1, 1), filters_out * mlp_mult, filters_out, bn=False)) if use_mlp else 0

    ct_block_params = q_conv_params + kv_conv_params + bn_params + conv_out + mlp_params

    return embed_conv_params + layer_norm_params + ct_block_params * blocks


def compute_op_params(op: str, input_shape: TensorShape, output_shape: TensorShape, op_regex_dict: 'dict[str, re.Pattern]'):
    preserve_shape = input_shape == output_shape
    # this operators can't have parameters in any case
    no_params_operators = ['add', 'concat', 'input', 'gap']

    input_filters = input_shape.c
    output_filters = output_shape.c

    if op in no_params_operators:
        return 0

    # if identity need to change shape, then it becomes a pointwise convolution
    if op == 'identity':
        return 0 if preserve_shape else compute_conv_params((1, 1), input_filters, output_filters)

    if op == 'lstm':
        return compute_lstm_params(input_filters, output_filters)

    if op == 'gru':
        return compute_gru_params(input_filters, output_filters)

    # single convolution case
    match = op_regex_dict['conv'].match(op)  # type: re.Match
    if match:
        return compute_conv_params(match.group('kernel'), input_filters, output_filters)

    # pooling case, if it needs to change shape, then it is followed by a pointwise convolution
    match = op_regex_dict['pool'].match(op)  # type: re.Match
    if match:
        return 0 if preserve_shape else compute_conv_params((1, 1), input_filters, output_filters)

    # stacked convolution case, both use batch normalization and only the first one modifies the shape
    match = op_regex_dict['stack_conv'].match(op)  # type: re.Match
    if match:
        return compute_conv_params(match.group('kernel_1'), input_filters, output_filters) + \
               compute_conv_params(match.group('kernel_2'), output_filters, output_filters)

    # depthwise separable convolution case
    match = op_regex_dict['dconv'].match(op)  # type: re.Match
    if match:
        return compute_dconv_params(match.group('kernel'), input_filters, output_filters)

    match = op_regex_dict['tconv'].match(op)  # type: re.Match
    if match:
        return compute_conv_params(match.group('kernel'), input_filters, output_filters) + \
               compute_conv_params(match.group('kernel'), output_filters, output_filters)

    # Convolutional vision transformer cases
    match = op_regex_dict['cvt'].match(op)  # type: re.Match
    if match:
        k, heads, blocks = int(match.group(1)), int(match.group(2)), int(match.group(3))
        return compute_cvt_params((k, k), heads, blocks, input_filters, output_filters, use_mlp=True)

    match = op_regex_dict['scvt'].match(op)  # type: re.Match
    if match:
        k, heads = int(match.group(1)), int(match.group(2))
        return compute_cvt_params((k, k), heads, 1, input_filters, output_filters, use_mlp=False)

    raise AttributeError(f'Unsupported operator "{op}"')


def compute_target_shapes(input_shape: TensorShape, cells_count: int, filters: int, reduction_cell_indices: 'list[int]') -> 'list[TensorShape]':
    output_shapes = []

    # replace last dimension with the filters
    input_shape = TensorShape(input_shape.h, input_shape.w, filters)
    output_shape = input_shape

    for cell_index in range(cells_count):
        if cell_index in reduction_cell_indices:
            output_shape = TensorShape(*(math.ceil(a * b) for a, b in zip(output_shape, reduction_shape_transform)))

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
            'block_index': [-1] * 2,
            'connect_lookback': [-100, -100],
            'params': [0, compute_conv_params((1, 1), concat_shape.c, output_shape.c)]
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


def merge_graphs(main_g: Graph, cell_g: Graph, lookback_indexes: 'list[Union[int, str]]', lookback_shapes: 'list[TensorShape]',
                 lookback_reshape: bool, output_shape: TensorShape, cell_index: int, op_regex_dict: 'dict[str, re.Pattern]'):
    # if shapes differ (and are not related to original input), then -2 lookback must be reshaped to the same dimension of -1 lookback
    # only height is checked since only one dimension is necessary to diverge to trigger the reshaping
    # avoid it in case lookback_reshape is set to false in configuration
    if lookback_reshape and (not lookback_shapes[-1].h == lookback_shapes[-2].h and not lookback_indexes[-2] == 'input'):
        layer_name = f'rs_{lookback_indexes[-2]}'

        main_g.add_vertex(layer_name, op='1x1 conv', params=compute_conv_params((1, 1), lookback_shapes[-2].c, lookback_shapes[-1].c),
                          cell_index=cell_index, block_index=-1)
        main_g.add_edge(lookback_indexes[-2], layer_name,
                        tensor_h=lookback_shapes[-2].h, tensor_w=lookback_shapes[-2].w, tensor_c=lookback_shapes[-2].c)

        lookback_indexes[-2] = layer_name
        lookback_shapes[-2] = lookback_shapes[-1]

    lb1_vs = cell_g.vs.select(connect_lookback=-1)
    lb2_vs = cell_g.vs.select(connect_lookback=-2)
    internal_vs = cell_g.vs.select(connect_lookback_ge=0)

    # compute parameters of all operations of the cell
    lb1_vs['params'] = [compute_op_params(op, lookback_shapes[-1], output_shape, op_regex_dict) for op in lb1_vs['op']]
    lb2_vs['params'] = [compute_op_params(op, lookback_shapes[-2], output_shape, op_regex_dict) for op in lb2_vs['op']]
    internal_vs['params'] = [compute_op_params(op, output_shape, output_shape, op_regex_dict) for op in internal_vs['op']]

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

    It is also useful to estimate quickly the amount of memory required by the neural network.
    '''

    def __init__(self, cell_spec: list, input_shape: 'tuple[int, ...]', filters: int, num_classes: int,
                 motifs: int, normals_per_motif: int, lookback_reshape: bool, op_regex_dict: 'dict[str, re.Pattern]') -> None:
        super().__init__()
        self.op_regex_dict = op_regex_dict

        flat_inputs = list_flatten(cell_spec)[::2]
        cells_count = motifs * (normals_per_motif + 1) - 1

        used_lookbacks = set(filter(lambda el: el < 0, flat_inputs))
        # use_skips = any(x < -1 for x in used_lookbacks)
        used_cell_indices = list(range(cells_count - 1, -1, max(used_lookbacks, default=cells_count)))

        reduction_cell_indices = list(range(normals_per_motif, cells_count, normals_per_motif + 1))

        if len(input_shape) <= 3:
            input_shape = TensorShape(*input_shape) if len(input_shape) == 3 else TensorShape(input_shape[0], 1, input_shape[1])
        else:
            raise ValueError(f'Too much input dimensions ({len(input_shape)}), not supported by graph generator')

        target_shapes = compute_target_shapes(input_shape, cells_count, filters, reduction_cell_indices)

        # initialize the graph with only the input node
        g = Graph(n=1, directed=True)
        g.vs['op'] = ['input']
        g.vs['params'] = [0]
        g.vs['cell_index'] = [0]
        g.vs['block_index'] = [-1]
        g.vs['name'] = ['input']

        # use integer indices initially, then it's easier to use the names (select support index or name (string vertex attribute))
        lookback_vertex_indices = [0, 0]
        lookback_shapes = [input_shape, input_shape]

        # iterate on cells, using reduction bool list since we need the value later
        for cell_index, output_shape in enumerate(target_shapes):
            if cell_index in used_cell_indices:
                cell_g = create_cell_graph(cell_spec, cell_index, output_shape)
                g = merge_graphs(g, cell_g, lookback_vertex_indices, lookback_shapes, lookback_reshape, output_shape, cell_index, op_regex_dict)

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
            'cell_index': [cells_count - 1] * 2,
            'block_index': [-1] * 2,
            'params': [0, num_classes * (output_shape.c + 1)]
        }
        g.add_vertices(2, attributes=v_attributes)

        # TODO: adapt t
        # connect last cell to GAP and GAP to Softmax
        edge_attributes = {
            'tensor_h': [1, 1],
            'tensor_w': [1, 1],
            'tensor_c': [output_shape.c, num_classes]
        }
        g.add_edges([(lookback_vertex_indices[-1], 'GAP'), ('GAP', 'Softmax')], attributes=edge_attributes)

        # print(g)
        self.g = g
        self.cells_count = cells_count

    def get_total_params(self):
        return sum(self.g.vs['params'])

    def get_params_per_cell(self):
        return [sum(self.g.vs.select(cell_index=i)['params']) for i in range(0, self.cells_count)]

    def get_params_up_through_cell_index(self, c_index: int):
        ''' Cell index is inclusive. '''
        return sum(self.g.vs.select(cell_index_lt=c_index)['params'])

    def get_max_cells_cut_under_param_constraint(self, params_constraint: int):
        '''
        Returns the maximum number of cells that can be taken in a cut, starting from first cell,
        that satisfies the parameter constraint (memory).
        '''
        params_per_cell = self.get_params_per_cell()

        for i in range(self.cells_count, 0, -1):
            if sum(params_per_cell[:i]) <= params_constraint:
                return i

        return 0
