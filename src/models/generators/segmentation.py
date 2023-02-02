from typing import Type

import tensorflow as tf
from tensorflow.keras import layers, metrics, Model, losses

from models.generators.base import BaseModelGenerator, NetworkBuildInfo, WrappedTensor
from models.results import BaseTrainingResults, SegmentationTrainingResults
from search_space import CellSpecification
from utils.func_utils import elementwise_mult


class SegmentationModelGenerator(BaseModelGenerator):
    '''
    Model generator for image semantic segmentation tasks.
    The generated models have a U-Net like structure,
    but each level of the encoder-decoder structure is composed of cells found during the NAS process.
    '''

    def get_maximum_cells(self):
        encoder_cells = self.motifs * (self.normal_cells_per_motif + 1) - 1
        decoder_cells = (self.motifs - 1) * (self.normal_cells_per_motif + 1)

        return encoder_cells + decoder_cells

    def _generate_network_info(self, cell_spec: CellSpecification, use_stem: bool) -> NetworkBuildInfo:
        blocks = len(cell_spec)

        cell_inputs = cell_spec.inputs()
        # take only BLOCK input indexes (list even indices, discard -1 and -2), eliminating duplicates
        used_block_outputs = set(filter(lambda el: el >= 0, cell_inputs))
        used_lookbacks = set(filter(lambda el: el < 0, cell_inputs))
        unused_block_outputs = [x for x in range(0, blocks) if x not in used_block_outputs]
        use_skip = used_lookbacks.issuperset({-2})

        # additional info regarding the cell stack
        max_lookback = max(used_lookbacks, default=1)
        used_cell_indexes = [index for index in list(range(self.get_maximum_cells()))[::max_lookback]]

        encoder_cells = self.motifs * (self.normal_cells_per_motif + 1) - 1
        reduction_cell_indexes = list(range(self.normal_cells_per_motif, encoder_cells, self.normal_cells_per_motif + 1))
        need_input_norm_indexes = [el - min(used_lookbacks) - 1 for el in reduction_cell_indexes] if use_skip else []

        return NetworkBuildInfo(cell_spec, blocks, used_lookbacks, unused_block_outputs, use_skip, used_cell_indexes,
                                reduction_cell_indexes, need_input_norm_indexes)

    def _compute_cell_output_shapes(self) -> 'list[list[int, ...]]':
        output_shapes = []
        current_shape = [1] * self.op_dims + [self.filters_ratio]
        reduction_tx = [0.5] * self.op_dims + [2]
        upsample_tx = [2.0] * self.op_dims + [0.5]

        # ENCODER shapes
        for motif_index in range(self.motifs):
            # add N times a normal cell
            for _ in range(self.normal_cells_per_motif):
                output_shapes.append(current_shape)

            # add 1 time a reduction cell, except for the last motif
            if motif_index + 1 < self.motifs:
                current_shape = elementwise_mult(current_shape, reduction_tx)
                output_shapes.append(current_shape)

        # DECODER shapes
        for motif_index in range(self.motifs - 1):
            # upsample cell
            current_shape = elementwise_mult(current_shape, upsample_tx)
            output_shapes.append(current_shape)

            # add N times a normal cell
            for _ in range(self.normal_cells_per_motif):
                output_shapes.append(current_shape)

        return output_shapes

    def _generate_output(self, input_tensor: tf.Tensor, dropout_prob: float = 0.0) -> tf.Tensor:
        # don't add suffix in models with a single output
        name_suffix = f'_c{self.cell_index}' if self.cell_index < self.get_maximum_cells() else ''

        if dropout_prob > 0.0:
            input_tensor = layers.Dropout(dropout_prob)(input_tensor)

        output = self.op_instantiator.generate_pointwise_conv(self.output_classes_count, strided=False, activation_f=tf.nn.softmax,
                                                              name=f'Pointwise_Softmax{name_suffix}')(input_tensor)
        # TODO: need upscaling (nearest neighbour) in case of intermediate outputs

        self.output_layers.update({f'Pointwise_Softmax{name_suffix}': output})
        return output

    def build_model(self, cell_spec: CellSpecification, add_imagenet_stem: bool = False) -> 'tuple[Model, list[str]]':
        # stores the final cell's output of each U-net structure "level", the ones used in the skip connections.
        level_outputs = []

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
            filters = filters // 2
            # generate an upsample cell any time the model creates a new decoder level
            cell_inputs = self._build_upsample_cell(cell_inputs, level_outputs, filters)

            # add N times a normal cell
            for _ in range(N):
                cell_inputs = self._build_cell_util(filters, cell_inputs)

        # take last cell output and use it in the final output
        last_output = cell_inputs[-1]

        model = self._finalize_model(model_input, last_output.tensor)
        return model, self._get_output_names()

    def _build_upsample_unit(self, lookback_inputs: 'list[WrappedTensor]', level_outputs: 'list[WrappedTensor]',
                             filters: int) -> 'list[WrappedTensor]':
        '''
        Generate an upsample unit to alter the lookback inputs. The upsample unit structure is fixed and therefore not searched during NAS.

        The -1 lookback input is processed using a linear upsample, then concatenated with the specular output of the encoder part (see U-net).
        The concatenation is processed by a pointwise convolution to adapt the filters, finalizing -1 lookback generation.

        The -2 lookback is instead generated using only a transpose convolution of the original -2 lookback.

        NOTE: the last level output is popped from the list, so this function contains a side effect, but it simplifies the logic and return.

        Args:
            lookback_inputs: original lookback inputs.
            level_outputs: U-net encoder outputs.
            filters: target number of filters for the next cell.

        Returns:
            the new lookback inputs, used in the upsample cell.
        '''
        target_shape = self._current_target_shape()
        lb2, lb1 = lookback_inputs

        # build -1 lookback input, as a linear upsample + concatenation with specular output of last encoder layer on the same U-net "level".
        # the number of filters is regularized with a pointwise convolution.
        if lb1 is not None:
            linear_upsampled_tensor = self.op_instantiator.generate_linear_upsample(2, name=f'lin_upsample_unit_c{self.cell_index}')(lb1.tensor)
            specular_output = level_outputs.pop().tensor
            unet_concat = layers.Concatenate(name=f'u_net_concat_c{self.cell_index}')([linear_upsampled_tensor, specular_output])
            # TODO: the compression is not necessary since operators could directly adapt the depth dimension,
            #  but using it should reduce quite a bit the number of parameters and FLOPs performed by the upsample cell operators.
            lb1_tensor = self.op_instantiator.generate_pointwise_conv(filters, strided=False,
                                                                      name=f'up_pointwise_filter_compressor_c{self.cell_index}')(unet_concat)
            lb1 = WrappedTensor(lb1_tensor, target_shape)

        # build -2 lookback input as an upsample of the original -2 lookback, with the use of a transpose convolution
        if lb2 is not None:
            lb2_tensor = self.op_instantiator.generate_transpose_conv(filters, 2, name=f'transpose_conv_upsample_unit_c{self.cell_index}')(lb2.tensor)
            lb2 = WrappedTensor(lb2_tensor, target_shape)

        return [lb2, lb1]

    def _build_upsample_cell(self, lookback_inputs: 'list[WrappedTensor]', level_outputs: 'list[WrappedTensor]',
                             filters: int) -> 'list[WrappedTensor]':
        target_output_shape = self._current_target_shape()
        upsampled_lookbacks = self._build_upsample_unit(lookback_inputs, level_outputs, filters)
        new_lookbacks = self._build_cell_util(filters, upsampled_lookbacks)

        # as -2 lookback, keep the original -1 lookback upsampled through transpose convolution, instead of the "concat with encoder level" tensor
        _, lb1 = lookback_inputs
        if lb1 is not None:
            lb2 = self.op_instantiator.generate_transpose_conv(filters, 2, name=f'transpose_lb2_upsample_c{self.cell_index}')(lb1.tensor)
        else:
            lb2 = None

        return [WrappedTensor(lb2, target_output_shape), new_lookbacks[-1]]

    def _get_loss_function(self) -> losses.Loss:
        return losses.CategoricalCrossentropy()

    def _get_metrics(self) -> 'list[metrics.Metric]':
        return ['accuracy', metrics.OneHotMeanIoU(self.output_classes_count, name='mean_iou')]

    @staticmethod
    def get_results_processor_class() -> Type[BaseTrainingResults]:
        return SegmentationTrainingResults
