from typing import Type

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers, Model, losses, metrics

from models.generators.base import BaseModelGenerator, NetworkBuildInfo, WrappedTensor
from models.graphs.network_graph import NetworkGraph
from models.results import BaseTrainingResults, ClassificationTrainingResults
from search_space import CellSpecification
from utils.func_utils import elementwise_mult


class ClassificationModelGenerator(BaseModelGenerator):
    '''
    Model generator for image and time series classification tasks.
    The generated models have a similar structure to the one defined by PNAS and NASNet.
    '''

    def _generate_network_info(self, cell_spec: CellSpecification, use_stem: bool) -> NetworkBuildInfo:
        blocks = len(cell_spec)
        total_cells = self.get_maximum_cells() + 2 if use_stem else self.get_maximum_cells()

        use_skip = -2 in cell_spec.used_lookbacks
        nearest_lookback = max(cell_spec.used_lookbacks, default=1)
        # "nearest_lookback" is negative, so the slice correctly starts from the last cell (which is the one connected to the output)
        used_cell_indexes = list(range(self.get_maximum_cells()))[::nearest_lookback]

        # additional info regarding the cell stack, with stem the logic is similar but dividing the two cases make the code more clear
        # TODO: actually, this is a mess, refactor this if stem is kept in the future
        if use_stem:
            reduction_cell_indexes = [0, 1] + list(range(2 + self.normal_cells_per_motif, total_cells, self.normal_cells_per_motif + 1))
            need_input_norm_indexes = [0] + [el - min(cell_spec.used_lookbacks) - 1 for el in reduction_cell_indexes] \
                if use_skip and self.lookback_reshape else []
        else:
            reduction_cell_indexes = list(range(self.normal_cells_per_motif, total_cells, self.normal_cells_per_motif + 1))
            need_input_norm_indexes = [el - min(cell_spec.used_lookbacks) - 1 for el in reduction_cell_indexes] \
                if use_skip and self.lookback_reshape else []

        return NetworkBuildInfo(cell_spec, blocks, used_cell_indexes, reduction_cell_indexes, need_input_norm_indexes)

    def _compute_cell_output_shapes(self) -> 'list[list[int, ...]]':
        output_shapes = []
        current_shape = [1] * self.op_dims + [self.filters_ratio]
        reduction_tx = [0.5] * self.op_dims + [2]

        for motif_index in range(self.motifs):
            # add N times a normal cell
            for _ in range(self.normal_cells_per_motif):
                output_shapes.append(current_shape)

            # add 1 time a reduction cell, except for the last motif
            if motif_index + 1 < self.motifs:
                current_shape = elementwise_mult(current_shape, reduction_tx)
                output_shapes.append(current_shape)

        return output_shapes

    def get_output_layers_name(self) -> str:
        return 'Softmax'

    def _generate_output(self, hidden_tensor: WrappedTensor, dropout_prob: float = 0.0) -> tf.Tensor:
        # don't add suffix in models with a single output
        name_suffix = f'_c{self.cell_index}' if self.cell_index < self.get_maximum_cells() else ''

        gap = self.op_instantiator.gap(name=f'GAP{name_suffix}')(hidden_tensor.tensor)
        if dropout_prob > 0.0:
            gap = layers.Dropout(dropout_prob)(gap)

        output_name = self.get_output_layers_name()
        output = layers.Dense(self.output_classes_count, activation='softmax', kernel_regularizer=self.l2_weight_reg,
                              name=f'{output_name}{name_suffix}')(gap)

        self.output_layers.update({f'{output_name}{name_suffix}': output})
        return output

    def build_model_graph(self, cell_spec: CellSpecification, add_imagenet_stem: bool = False) -> NetworkGraph:
        net_info = self._generate_network_info(cell_spec, use_stem=False)
        net = NetworkGraph(list(self.input_shape), self.op_instantiator, cell_spec, self.cell_output_shapes,
                           net_info.used_cell_indexes, net_info.need_input_norm_indexes, self.residual_cells)

        M, N = self._get_macro_params(cell_spec)
        for m in range(M):
            # add N normal cell per motif
            for _ in range(N):
                net.build_cell()

            # add reduction cell, except for the last motif
            if m + 1 < M:
                net.build_cell()

        # add output layers
        final_cell_out = net.lookback_nodes[-1]
        output_filters = final_cell_out.shape[-1]
        v_attributes = {
            'name': ['GAP', 'dense_softmax'],
            'op': ['gap', 'dense'],
            'cell_index': [-1] * 2,
            'block_index': [-1] * 2,
            'params': [0, self.output_classes_count * (output_filters + 1)]
        }
        net.g.add_vertices(2, v_attributes)

        # add edges from inputs to operator layers, plus from operator layers to add
        edges = [(final_cell_out.name, 'GAP'), ('GAP', 'dense_softmax')]
        edge_attributes = {
            'tensor_shape': [str(final_cell_out.shape), str([output_filters])]
        }
        net.g.add_edges(edges, edge_attributes)

        return net

    def build_model(self, cell_spec: CellSpecification, add_imagenet_stem: bool = False) -> 'tuple[Model, list[str]]':
        self.network_build_info = self._generate_network_info(cell_spec, add_imagenet_stem)
        self.output_layers = {}

        M, N = self._get_macro_params(cell_spec)
        filters = self.filters
        self._reset_metadata_indexes()

        model_input = layers.Input(shape=self.input_shape)
        initial_lookback_tensor = self._apply_preprocessing_and_augmentation(model_input)
        initial_shape_ratios = [1] * len(self.input_shape)
        initial_lookback_input = WrappedTensor(initial_lookback_tensor, initial_shape_ratios)
        # define inputs usable by blocks, both set to input image (or preprocessed / data augmentation of the input) at start
        cell_inputs = [initial_lookback_input, initial_lookback_input]  # [skip, last]

        if add_imagenet_stem:
            cell_inputs, filters = self._prepend_imagenet_stem(cell_inputs, filters)

        # add M times N normal cells and a reduction cell (except in the last motif where the reduction cell is skipped)
        for motif_index in range(M):
            # add N times a normal cell
            for _ in range(N):
                cell_inputs = self._build_cell_util(filters, cell_inputs)

            # add 1 time a reduction cell, except for the last motif
            if motif_index + 1 < M:
                filters = filters * 2
                cell_inputs = self._build_cell_util(filters, cell_inputs, reduction=True)

        # take last cell output and use it in GAP
        last_output = cell_inputs[-1]

        model = self._finalize_model(model_input, last_output)
        return model, self._get_output_names()

    def _get_loss_function(self) -> losses.Loss:
        return losses.CategoricalCrossentropy()

    def _get_metrics(self) -> 'list[metrics.Metric]':
        return ['accuracy', tfa.metrics.F1Score(num_classes=self.output_classes_count, average='macro'), metrics.TopKCategoricalAccuracy(k=5)]

    @staticmethod
    def get_results_processor_class() -> Type[BaseTrainingResults]:
        return ClassificationTrainingResults
