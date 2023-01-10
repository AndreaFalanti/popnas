import math
from typing import Type

import tensorflow as tf
from tensorflow.keras import layers, metrics, Model, losses

from models.generators.base import BaseModelGenerator, NetworkBuildInfo
from models.results import BaseTrainingResults, SegmentationTrainingResults
from utils.func_utils import list_flatten


class SegmentationModelGenerator(BaseModelGenerator):
    '''
    Model generator for image and time series classification tasks.
    The generated models have a similar structure to the one defined by PNAS and NASNet.
    '''

    def get_maximum_cells(self):
        encoder_cells = self.motifs * (self.normal_cells_per_motif + 1) - 1
        decoder_cells = (self.motifs - 1) * self.normal_cells_per_motif

        return encoder_cells + decoder_cells

    def get_real_cell_depth(self, cell_spec: list):
        used_lookbacks = set(filter(lambda el: el < 0, list_flatten(cell_spec)[::2]))

        if -1 in used_lookbacks:
            return self.get_maximum_cells()
        else:
            encoder_cells = self.motifs * (self.normal_cells_per_motif + 1) - 1
            # assign -1 to upsample units, which are discarded from actual count of cells
            architecture_indexes = list(range(encoder_cells)) + \
                                   [val for c_index in range(encoder_cells, encoder_cells + self.motifs - 1) for val in (-1, c_index)]

            # additional info regarding the cell stack
            return len([index for index in architecture_indexes if index != -1])

    def _generate_network_info(self, cell_spec: 'list[tuple]', use_stem: bool) -> NetworkBuildInfo:
        # it's a list of tuples, so already grouped by 4
        blocks = len(cell_spec)

        flat_cell_spec = list_flatten(cell_spec)
        # take only BLOCK input indexes (list even indices, discard -1 and -2), eliminating duplicates
        used_block_outputs = set(filter(lambda el: el >= 0, flat_cell_spec[::2]))
        used_lookbacks = set(filter(lambda el: el < 0, flat_cell_spec[::2]))
        unused_block_outputs = [x for x in range(0, blocks) if x not in used_block_outputs]
        use_skip = used_lookbacks.issuperset({-2})

        # utility array for next computations, upsample units are not cells, but still considered for skip connections
        # the first list is the encoder part, the second list refers to the decoder part
        # -1 refers to upsample units and should be discarded
        encoder_cells = self.motifs * (self.normal_cells_per_motif + 1) - 1
        architecture_indexes = list(range(encoder_cells)) + \
                               [val for c_index in range(encoder_cells, encoder_cells + self.motifs - 1) for val in (-1, c_index)]

        # additional info regarding the cell stack
        used_cell_indexes = [index for index in architecture_indexes if index != -1]
        reduction_cell_indexes = list(range(self.normal_cells_per_motif, encoder_cells, self.normal_cells_per_motif + 1))
        need_input_norm_indexes = [el - min(used_lookbacks) - 1 for el in reduction_cell_indexes] if use_skip else []

        return NetworkBuildInfo(cell_spec, blocks, used_lookbacks, unused_block_outputs, use_skip, used_cell_indexes,
                                reduction_cell_indexes, need_input_norm_indexes)

    def _compute_cell_output_shapes(self) -> 'list[list[int, ...]]':
        output_shapes = []
        current_shape = list(self.input_shape)
        current_shape[-1] = self.filters
        reduction_tx = [0.5] * self.op_dims + [2]
        upsample_tx = [2.0] * self.op_dims + [0.5]

        # ENCODER shapes
        for motif_index in range(self.motifs):
            # add N times a normal cell
            for _ in range(self.normal_cells_per_motif):
                output_shapes.append(current_shape)

            # add 1 time a reduction cell, except for the last motif
            if motif_index + 1 < self.motifs:
                current_shape = [math.ceil(val * tx) for val, tx in zip(current_shape, reduction_tx)]
                output_shapes.append(current_shape)

        # DECODER shapes
        for motif_index in range(self.motifs - 1):
            # upsample
            current_shape = [math.ceil(val * tx) for val, tx in zip(current_shape, upsample_tx)]

            # add N times a normal cell
            for _ in range(self.normal_cells_per_motif):
                output_shapes.append(current_shape)

        return output_shapes

    def _generate_output(self, input_tensor: tf.Tensor, dropout_prob: float = 0.0) -> tf.Tensor:
        # don't add suffix in models with a single output
        name_suffix = f'_c{self.cell_index}' if self.cell_index < self.total_cells else ''

        if dropout_prob > 0.0:
            input_tensor = layers.Dropout(dropout_prob)(input_tensor)

        output = self.op_instantiator.generate_pointwise_conv(self.output_classes_count, strided=False, activation_f=tf.nn.softmax,
                                                              name=f'Pointwise_Softmax{name_suffix}')(input_tensor)
        # TODO: need upscaling (nearest neighbour) in case of intermediate outputs

        self.output_layers.update({f'Pointwise_Softmax{name_suffix}': output})
        return output

    def build_model(self, cell_spec: 'list[tuple]', add_imagenet_stem: bool = False) -> 'tuple[Model, list[str]]':
        # stores the final cell's output of each U-net structure "level", the ones used in the skip connections.
        level_outputs = []

        self.network_build_info = self._generate_network_info(cell_spec, add_imagenet_stem)
        self.output_layers = {}

        M, N = self._get_macro_params(cell_spec)
        filters = self.filters
        self._reset_metadata_indexes()

        model_input = layers.Input(shape=self.input_shape)
        initial_lookback_input = self._apply_preprocessing_and_augmentation(model_input)
        # define inputs usable by blocks, both set to input image (or preprocessed / data augmentation of the input) at start
        cell_inputs = [initial_lookback_input, initial_lookback_input]  # [skip, last]

        if add_imagenet_stem:
            self._logger.warn('ImageNet stem is not supported in segmentation networks, the argument is ignored.')

        # ENCODER part of the U-net structure
        # add M times N normal cells and a reduction cell (except in the last motif where the reduction cell is skipped)
        for motif_index in range(M):
            # add N times a normal cell
            for _ in range(N):
                cell_inputs = self._build_cell_util(filters, cell_inputs)

            # add 1 time a reduction cell, except for the last motif
            # also save the encoder level output, so that it can be used in skip connections on the same level of the decoder
            if motif_index + 1 < M:
                level_outputs.append(cell_inputs[-1])
                filters = filters * 2
                cell_inputs = self._build_cell_util(filters, cell_inputs, reduction=True)

        # DECODER part of the U-net structure
        # add (M - 1) times an upsample unit followed by N normal cells
        for motif_index in range(M - 1):
            # generate new lookback input with the upsample unit
            cell_inputs = self._build_upsample_unit(cell_inputs, level_outputs, filters)

            # add N times a normal cell
            for _ in range(N):
                cell_inputs = self._build_cell_util(filters, cell_inputs)

        # take last cell output and use it in the final output
        last_output = cell_inputs[-1]

        model = self._finalize_model(model_input, last_output)
        output_names = list(self.output_layers.keys())
        return model, output_names

    def _build_upsample_unit(self, cell_inputs: 'list[tf.Tensor]', level_outputs: 'list[tf.Tensor]', filters: int) -> 'list[tf.Tensor]':
        '''
        Generate an upsample unit to alter the lookback inputs. The upsample unit structure is fixed and therefore not searched during NAS.
        The first valid lookback input is processed to generate the first new lookback, using a linear upsample,
        then concatenated with the specular output of the encoder part (see U-net). The concatenation is processed by a pointwise convolution
        to adapt the filters. The second lookback is instead generated using only a transpose convolution.

        NOTE: the last level output is popped from the list, so this function contains a side effect, but it simplifies the logic and return.

        Args:
            cell_inputs: original lookback inputs.
            level_outputs: U-net encoder outputs.
            filters: target number of filters for the next cell.

        Returns:
            the new lookback inputs.
        '''
        # scan the reversed list of lookback inputs, checking for the first not None tensor (nearest lookback usable).
        valid_input = next(inp for inp in cell_inputs[::-1] if inp is not None)

        # build -1 lookback input, as a linear upsample + concatenation with specular output of last encoder layer on the same U-net "level".
        # the number of filters is regularized with a pointwise convolution.
        linear_upsampled_tensor = self.op_instantiator.generate_linear_upsample(2, name=f'lin_upsample_c{self.cell_index}')(valid_input)
        specular_output = level_outputs.pop()
        upsample_specular_concat = layers.Concatenate()([linear_upsampled_tensor, specular_output])
        lb1 = self.op_instantiator.generate_pointwise_conv(filters, strided=False,
                                                           name=f'up_pointwise_filter_compressor_c{self.cell_index}')(upsample_specular_concat)

        # build -2 lookback input
        lb2 = self.op_instantiator.generate_transpose_conv(filters, 2, name=f'transpose_conv_upsample_c{self.cell_index}')(valid_input)

        return [lb2, lb1]

    def _get_loss_function(self) -> losses.Loss:
        return losses.CategoricalCrossentropy()

    def _get_metrics(self) -> 'list[metrics.Metric]':
        return ['accuracy', metrics.MeanIoU(self.output_classes_count)]

    def get_results_processor_class(self) -> Type[BaseTrainingResults]:
        return SegmentationTrainingResults
