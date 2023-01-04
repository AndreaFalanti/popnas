import math
from typing import Type

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers, Model, losses, metrics

from models.generators.base import BaseModelGenerator, NetworkBuildInfo
from models.results import BaseTrainingResults, ClassificationTrainingResults
from utils.func_utils import list_flatten


class ClassificationModelGenerator(BaseModelGenerator):
    '''
    Model generator for image and time series classification tasks.
    The generated models have a similar structure to the one defined by PNAS and NASNet.
    '''

    def _generate_network_info(self, cell_spec: 'list[tuple]', use_stem: bool) -> NetworkBuildInfo:
        # it's a list of tuples, so already grouped by 4
        blocks = len(cell_spec)

        flat_cell_spec = list_flatten(cell_spec)
        # take only BLOCK input indexes (list even indices, discard -1 and -2), eliminating duplicates
        used_block_outputs = set(filter(lambda el: el >= 0, flat_cell_spec[::2]))
        used_lookbacks = set(filter(lambda el: el < 0, flat_cell_spec[::2]))
        unused_block_outputs = [x for x in range(0, blocks) if x not in used_block_outputs]
        use_skip = used_lookbacks.issuperset({-2})

        # additional info regarding the cell stack, with stem the logic is similar but dividing the two cases make the code more clear
        if use_stem:
            total_cells = self.total_cells + 2
            used_cell_indexes = list(range(total_cells - 1, -1, max(used_lookbacks, default=total_cells)))
            reduction_cell_indexes = [0, 1] + list(range(2 + self.normal_cells_per_motif, total_cells, self.normal_cells_per_motif + 1))
            need_input_norm_indexes = [0] + [el - min(used_lookbacks) - 1 for el in reduction_cell_indexes] if use_skip else []
        else:
            used_cell_indexes = list(range(self.total_cells - 1, -1, max(used_lookbacks, default=self.total_cells)))
            reduction_cell_indexes = list(range(self.normal_cells_per_motif, self.total_cells, self.normal_cells_per_motif + 1))
            need_input_norm_indexes = [el - min(used_lookbacks) - 1 for el in reduction_cell_indexes] if use_skip else []

        return NetworkBuildInfo(cell_spec, blocks, used_lookbacks, unused_block_outputs, use_skip, used_cell_indexes,
                                reduction_cell_indexes, need_input_norm_indexes)

    def _compute_cell_output_shapes(self) -> 'list[list[int, ...]]':
        output_shapes = []
        current_shape = list(self.input_shape)
        current_shape[-1] = self.filters
        reduction_tx = [0.5] * self.op_dims + [2]

        for motif_index in range(self.motifs):
            # add N times a normal cell
            for _ in range(self.normal_cells_per_motif):
                output_shapes.append(current_shape)

            # add 1 time a reduction cell, except for last motif
            if motif_index + 1 < self.motifs:
                current_shape = [math.ceil(val * tx) for val, tx in zip(current_shape, reduction_tx)]
                output_shapes.append(current_shape)

        return output_shapes

    def _generate_output(self, input_tensor: tf.Tensor, dropout_prob: float = 0.0) -> tf.Tensor:
        # don't add suffix in models with a single output
        name_suffix = f'_c{self.cell_index}' if self.cell_index < self.total_cells else ''

        gap = self.op_instantiator.gap(name=f'GAP{name_suffix}')(input_tensor)
        if dropout_prob > 0.0:
            gap = layers.Dropout(dropout_prob)(gap)
        output = layers.Dense(self.output_classes_count, activation='softmax', kernel_regularizer=self.l2_weight_reg,
                              name=f'Softmax{name_suffix}')(gap)

        self.output_layers.update({f'Softmax{name_suffix}': output})
        return output

    def build_model(self, cell_spec: 'list[tuple]', add_imagenet_stem: bool = False) -> 'tuple[Model, dict, int]':
        self.network_build_info = self._generate_network_info(cell_spec, add_imagenet_stem)
        self.output_layers = {}
        # store partition sizes (computed between each two adjacent cells and between last cell and GAP)
        partitions_dict = {}

        M, N = self._get_macro_params(cell_spec)
        filters = self.filters
        self._reset_metadata_indexes()

        model_input = layers.Input(shape=self.input_shape)
        initial_lookback_input = self._apply_preprocessing_and_augmentation(model_input)
        # define inputs usable by blocks, both set to input image (or preprocessed / data augmentation of the input) at start
        cell_inputs = [initial_lookback_input, initial_lookback_input]  # [skip, last]

        if add_imagenet_stem:
            cell_inputs, filters = self._prepend_imagenet_stem(cell_inputs, filters, partitions_dict)

        # add M times N normal cells and a reduction cell (except in the last motif where the reduction cell is skipped)
        for motif_index in range(M):
            # add N times a normal cell
            for _ in range(N):
                cell_inputs = self._build_cell_util(filters, cell_inputs, partitions_dict)

            # add 1 time a reduction cell, except for the last motif
            if motif_index + 1 < M:
                filters = filters * 2
                cell_inputs = self._build_cell_util(filters, cell_inputs, partitions_dict, reduction=True)

        # take last cell output and use it in GAP
        last_output = cell_inputs[-1]

        model = self._finalize_model(model_input, last_output)
        last_cell_index = max(self.network_build_info.used_cell_indexes, default=0)
        return model, partitions_dict, last_cell_index

    def _get_loss_function(self) -> losses.Loss:
        return losses.CategoricalCrossentropy()

    def _get_metrics(self) -> 'list[metrics.Metric]':
        return ['accuracy', tfa.metrics.F1Score(num_classes=self.output_classes_count, average='macro')]

    def get_results_processor_class(self) -> Type[BaseTrainingResults]:
        return ClassificationTrainingResults
