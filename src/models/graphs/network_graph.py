from igraph import Graph

from models.graphs.units import TensorNode, build_cell_dag
from models.operators.op_instantiator import OpInstantiator
from search_space import CellSpecification


class NetworkGraph:
    def __init__(self, input_shape: 'list[float]', op_instantiator: OpInstantiator,
                 cell_specification: CellSpecification, target_shapes: 'list[list[float, ...]]',
                 used_cell_indexes: list[int], lookback_reshape_cell_indexes: list[int], residual_output: bool):
        self.cell_spec = cell_specification
        self.op_instantiator = op_instantiator
        self.residual_output = residual_output
        self.lookback_reshape_cell_indexes = lookback_reshape_cell_indexes
        self.used_cell_indexes = used_cell_indexes
        self.last_cell_index = max(used_cell_indexes)

        # target shapes are ratios of original input.
        # since the param computations need the actual number of filters, convert the ratio into the filter values.
        # spatial dimensions are kept as ratios (could be null in the actual input shape).
        input_filters = input_shape[-1]
        self.target_shapes = [s[:-1] + [input_filters * s[-1]] for s in target_shapes]

        # extract info from cell specification
        self.used_blocks = set(inp for inp in self.cell_spec.inputs() if inp >= 0)
        self.unused_blocks = [b for b in range(len(self.cell_spec)) if b not in self.used_blocks]
        self.nearest_used_lookback = max(inp for inp in self.cell_spec.inputs() if inp < 0)

        v_attributes = {
            'name': ['input'],
            'op': ['input'],
            'cell_index': [-1],
            'block_index': [-1],
            'params': [0]
        }
        self.g = Graph(n=1, directed=True, vertex_attrs=v_attributes)

        input_node = TensorNode('input', input_shape)
        self.lookback_nodes = [input_node, input_node]  # type: list[TensorNode]

        self.cell_index = 0

    def build_cell(self):
        if self.cell_index in self.used_cell_indexes:
            cell_target_shape = self.target_shapes[self.cell_index]
            cell_out_node = build_cell_dag(self.g, self.lookback_nodes, cell_target_shape, self.cell_spec,
                                           self.cell_index, self.residual_output, self.lookback_reshape_cell_indexes,
                                           self.op_instantiator.get_op_params)
            # update input lookbacks
            self.lookback_nodes = [self.lookback_nodes[-1], cell_out_node]
        else:
            self.lookback_nodes = [self.lookback_nodes[-1], None]

        self.cell_index = self.cell_index + 1

    def get_total_params(self):
        return sum(self.g.vs['params'])

    def get_params_per_cell(self):
        return [sum(self.g.vs.select(cell_index=i)['params']) for i in range(0, self.last_cell_index)]

    def get_params_per_layer(self):
        ''' Grouped by cell for more readability. '''
        return [list(zip(self.g.vs.select(cell_index=i)['name'], self.g.vs.select(cell_index=i)['params'])) for i in range(0, self.last_cell_index)]

    def get_params_up_through_cell_index(self, c_index: int):
        ''' Cell index is inclusive. '''
        return sum(self.g.vs.select(cell_index_lt=c_index)['params'])

    def get_max_cells_cut_under_param_constraint(self, params_constraint: int):
        '''
        Returns the maximum number of cells, starting from the first cell, which satisfies the parameter constraint (memory).
        If the network is cut after the cell index returned by this function, it will have fewer parameters than the constraint.
        '''
        params_per_cell = self.get_params_per_cell()

        for i in range(self.last_cell_index, 0, -1):
            if sum(params_per_cell[:i]) <= params_constraint:
                return i

        return 0
