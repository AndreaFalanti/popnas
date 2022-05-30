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

    def generate_network_graph(self, cell_spec: list):
        return NetworkGraph(cell_spec, self.input_shape, self.filters, self.num_classes, self.motifs, self.normals_per_motif)
