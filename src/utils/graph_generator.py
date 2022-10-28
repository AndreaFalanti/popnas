import re
from typing import Any

from utils.network_graph import NetworkGraph


class GraphGenerator:
    def __init__(self, cnn_config: 'dict[str, Any]', arc_config: 'dict[str, Any]',
                 input_shape: 'tuple[int, ...]', output_classes_count: int) -> None:
        super().__init__()

        self.input_shape = input_shape
        self.filters = cnn_config['filters']  # type: int
        self.num_classes = output_classes_count
        self.motifs = arc_config['motifs']  # type: int
        self.normals_per_motif = arc_config['normal_cells_per_motif']  # type: int
        self.lookback_reshape = arc_config['lookback_reshape']  # type: bool

        self.op_regex_dict = self.__compile_op_regexes()

    def __compile_op_regexes(self):
        '''
        Build a dictionary with compiled regexes for each parametrized supported operation.
        Adapt regexes based on input dimensionality.

        Returns:
            (dict): Regex dictionary
        '''
        # add groups to detect kernel size, based on op dimensionality.
        # e.g. Conv2D -> 3x3 conv, Conv1D -> 3 conv
        kernel_dims = len(self.input_shape) - 1
        op_kernel_group = 'x'.join([r'\d+'] * kernel_dims)
        opt_dilation_rate = rf'(:\d+dr)?'

        return {'conv': re.compile(rf'(?P<kernel>{op_kernel_group}){opt_dilation_rate} conv'),
                'dconv': re.compile(rf'(?P<kernel>{op_kernel_group}){opt_dilation_rate} dconv'),
                'tconv': re.compile(rf'(?P<kernel>{op_kernel_group}) tconv'),
                'stack_conv': re.compile(rf'(?P<kernel_1>{op_kernel_group})-(?P<kernel_2>{op_kernel_group}) conv'),
                'pool': re.compile(rf'(?P<kernel>{op_kernel_group}) (max|avg)pool'),
                'cvt': re.compile(r'(\d+)k-(\d+)h-(\d+)b cvt'),
                'scvt': re.compile(r'(\d+)k-(\d+)h scvt')}

    def generate_network_graph(self, cell_spec: list):
        return NetworkGraph(cell_spec, self.input_shape, self.filters, self.num_classes, self.motifs, self.normals_per_motif,
                            self.lookback_reshape, self.op_regex_dict)

    def alter_macro_structure(self, m: int, n: int, f: int):
        self.filters = f
        self.motifs = m
        self.normals_per_motif = n
