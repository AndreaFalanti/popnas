import math

import tensorflow as tf
from tensorflow.keras import layers, Model

import models.operators.layers as ops
from models.generators.base import BaseModelGenerator, NetworkBuildInfo
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

        if len(cell_spec) > 0:
            M = self.motifs
            N = self.normal_cells_per_motif
        # initial thrust case, empty cell
        else:
            M = 0
            N = 0

        # store partition sizes (computed between each two adjacent cells and between last cell and GAP)
        partitions_dict = {}

        filters = self.filters
        # reset indexes
        self.cell_index = 0
        self.block_index = 0
        self.prev_cell_index = 0

        model_input = layers.Input(shape=self.input_shape)

        start_lookback_input = model_input
        # some data preprocessing is integrated in the model. Update lookback inputs to be the output of the preprocessing model.
        if self.preprocessing_model is not None:
            start_lookback_input = self.preprocessing_model(start_lookback_input)
        # data augmentation integrated in the model to perform it in GPU. Update lookback inputs to be the output of the data augmentation model.
        if self.data_augmentation_model is not None:
            start_lookback_input = self.data_augmentation_model(start_lookback_input)

        # define inputs usable by blocks, both set to input image (or preprocessed / data augmentation of the input) at start
        cell_inputs = [start_lookback_input, start_lookback_input]  # [skip, last]

        # TODO: adapt it for 1D (time-series)
        # if stem is used, it will add a conv layer + 2 reduction cells at the start of the network
        if add_imagenet_stem:
            stem_conv = ops.Convolution(filters, kernel=(3, 3), strides=(2, 2), name='stem_3x3_conv', weight_reg=self.l2_weight_reg)(model_input)
            cell_inputs = [model_input, stem_conv]
            filters = filters * 2
            cell_inputs = self._build_cell_util(filters, cell_inputs, partitions_dict, reduction=True)
            filters = filters * 2
            cell_inputs = self._build_cell_util(filters, cell_inputs, partitions_dict, reduction=True)

        # add (M - 1) times N normal cells and a reduction cell
        for motif_index in range(M):
            # add N times a normal cell
            for _ in range(N):
                cell_inputs = self._build_cell_util(filters, cell_inputs, partitions_dict)

            # add 1 time a reduction cell, except for last motif
            if motif_index + 1 < M:
                filters = filters * 2
                cell_inputs = self._build_cell_util(filters, cell_inputs, partitions_dict, reduction=True)

        # take last cell output and use it in GAP
        last_output = cell_inputs[-1]

        # force to build output in case of initial thrust, since no cells are present (outputs are built in __build_cell_util if using multi-output)
        if self.multi_output and len(self.output_layers) > 0:
            return Model(inputs=model_input, outputs=self.output_layers.values()), partitions_dict, \
                   max(self.network_build_info.used_cell_indexes, default=0)
        else:
            output = self._generate_output(last_output, self.dropout_prob)
            return Model(inputs=model_input, outputs=output), partitions_dict, max(self.network_build_info.used_cell_indexes, default=0)
