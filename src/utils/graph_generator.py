import re
from typing import Any

from utils.network_graph import NetworkGraph


class GraphGenerator:
    def __init__(self, cnn_config: 'dict[str, Any]', arc_config: 'dict[str, Any]',
                 input_shape: 'tuple[int, int, int]', output_classes_count: int) -> None:
        super().__init__()

        self.input_shape = input_shape
        self.filters = cnn_config['filters']  # type: int
        self.num_classes = output_classes_count
        self.motifs = arc_config['motifs']  # type: int
        self.normals_per_motif = arc_config['normal_cells_per_motif']  # type: int

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
        op_kernel_groups = 'x'.join([r'(\d+)'] * kernel_dims)

        return {'conv': re.compile(rf'{op_kernel_groups} conv'),
                'dconv': re.compile(rf'{op_kernel_groups} dconv'),
                'tconv': re.compile(rf'{op_kernel_groups} tconv'),
                'stack_conv': re.compile(rf'{op_kernel_groups}-{op_kernel_groups} conv'),
                'pool': re.compile(rf'{op_kernel_groups} (max|avg)pool'),
                'cvt': re.compile(r'(\d+)k-(\d+)h-(\d+)b cvt'),
                'scvt': re.compile(r'(\d+)k-(\d+)h scvt')}

    def generate_network_graph(self, cell_spec: list):
        return NetworkGraph(cell_spec, self.input_shape, self.filters, self.num_classes,
                            self.motifs, self.normals_per_motif, self.op_regex_dict)
