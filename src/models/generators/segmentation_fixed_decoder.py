from typing import Type

import tensorflow as tf
from tensorflow.keras import layers, metrics, Model, losses

from models.generators.base import BaseModelGenerator, NetworkBuildInfo, WrappedTensor
from models.graphs.network_graph import NetworkGraph
from models.operators.layers import AtrousSpatialPyramidPooling, Convolution
from models.results import BaseTrainingResults, SegmentationTrainingResults
from search_space import CellSpecification
from utils.func_utils import elementwise_mult


class SegmentationFixedDecoderModelGenerator(BaseModelGenerator):
    '''
    Model generator for image semantic segmentation tasks.
    The generated models have a DeepLab like structure,
    where each level of the encoder structure is composed of cells found during the NAS process, while the decoder is fixed.

    NOTE: these models use bilinear upsample, which is not compilable with XLA (see: https://github.com/tensorflow/tensorflow/issues/57575).
    Do not set the XLA compilation flag in config, otherwise it will cause an error during model building!
    '''

    def _generate_network_info(self, cell_spec: CellSpecification, use_stem: bool) -> NetworkBuildInfo:
        blocks = len(cell_spec)
        use_skip = -2 in cell_spec.used_lookbacks
        total_cells = self.get_maximum_cells()

        # additional info regarding the cell stack
        nearest_lookback = max(cell_spec.used_lookbacks, default=1)
        # "nearest_lookback" is negative, so the slice correctly starts from the last cell (which is the one connected to the output)
        used_cell_indexes = list(range(total_cells))[::nearest_lookback]

        reduction_cell_indexes = list(range(self.normal_cells_per_motif, total_cells, self.normal_cells_per_motif + 1))
        need_input_norm_indexes = [el - min(cell_spec.used_lookbacks) - 1 for el in reduction_cell_indexes] if use_skip and self.lookback_reshape\
                else []

        return NetworkBuildInfo(cell_spec, blocks, used_cell_indexes, reduction_cell_indexes, need_input_norm_indexes)

    def _compute_cell_output_shapes(self) -> 'list[list[int, ...]]':
        output_shapes = []
        current_shape = [1] * self.op_dims + [self.filters_ratio]
        reduction_tx = [0.5] * self.op_dims + [2]

        # ENCODER shapes
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
        return 'pointwise_softmax'

    def _generate_output(self, hidden_tensor: WrappedTensor, dropout_prob: float = 0.0) -> tf.Tensor:
        # don't add suffix in models with a single output
        name_suffix = f'_c{self.cell_index}' if self.cell_index < self.get_maximum_cells() else ''

        # check the hidden state spatial ratio, since the resolution must match the initial input
        # intermediate representations needs to upscale the resolution by (1 / ratio)
        # since H and W have the same ratio, just check the first.
        hidden_tensor_spatial_ratio = hidden_tensor.shape[0]
        upscale_ratio = 1 / hidden_tensor_spatial_ratio

        h_tensor = hidden_tensor.tensor
        h_filters = int(hidden_tensor.shape[-1] * self.input_shape[-1])
        if upscale_ratio > 1:
            # h_tensor = self.op_instantiator.generate_transpose_conv(h_filters, int(upscale_ratio),
            #                                                         name=f'output_upsample{name_suffix}')(h_tensor)
            h_tensor = self.op_instantiator.generate_linear_upsample(int(upscale_ratio),
                                                                     name=f'output_upsample{name_suffix}')(h_tensor)

        # TODO: a spatial dropout could be more indicated for this structure
        if dropout_prob > 0.0:
            h_tensor = layers.Dropout(dropout_prob)(h_tensor)

        output_name = self.get_output_layers_name()
        output = self.op_instantiator.generate_pointwise_conv(self.output_classes_count, strided=False, activation_f=tf.nn.softmax,
                                                              name=f'{output_name}{name_suffix}')(h_tensor)

        self.output_layers.update({f'{output_name}{name_suffix}': output})
        return output

    def build_model_graph(self, cell_spec: CellSpecification, add_imagenet_stem: bool = False) -> NetworkGraph:
        net_info = self._generate_network_info(cell_spec, use_stem=False)
        net = NetworkGraph(list(self.input_shape), self.op_instantiator, cell_spec, self.cell_output_shapes,
                           net_info.used_cell_indexes, net_info.need_input_norm_indexes, self.residual_cells)

        M, N = self._get_macro_params(cell_spec)
        encoder_concat_nodes = []
        # encoder
        for m in range(M):
            for _ in range(N):
                net.build_cell()

            # add reduction cell, except for the last motif
            if m + 1 < M:
                encoder_concat_nodes.append(net.lookback_nodes[-1])
                net.build_cell()

        # TODO: decoder and output

        return net

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

            # add 1 time a reduction cell (and save encoder tensor for skip connections), except for the last motif
            if motif_index + 1 < M:
                # save the last produced cell output of the encoder level, so that it can be used in skip connections on the same level of the decoder
                last_encoder_output = next(t for t in cell_inputs[::-1] if t is not None)
                level_outputs.append(last_encoder_output)

                filters = filters * 2
                cell_inputs = self._build_cell_util(filters, cell_inputs, reduction=True)

        # apply spatial pyramid pooling on encoder bottleneck
        aspp = AtrousSpatialPyramidPooling(filters, dilation_rates=(3, 6, 12), weight_reg=self.l2_weight_reg)(cell_inputs[-1].tensor)
        encoder_out = WrappedTensor(aspp, cell_inputs[-1].shape)

        # DECODER part (fixed)
        # Inspired by DeepLab, upsample progressively with steps of factor 4
        if M > 0:
            reductions = M - 1
            upscale_factors = [4] * (reductions // 2)
            # when using an odd number of reductions, the last upsample will have a 2x upsample
            if reductions % 2 == 1:
                upscale_factors.append(2)

            # leave out the last upsample factor, which does not require a decoder unit
            decoder_out = encoder_out
            for i, upsample_factor in enumerate(upscale_factors[:-1]):
                encoder_level_pos = - 2 * (i + 1)
                encoder_t = level_outputs[encoder_level_pos]
                decoder_out = self.build_decoder_fixed_cell(upsample_factor, filters, decoder_out, encoder_t, i)

            # last_output_tensor = self.op_instantiator.generate_transpose_conv(filters, upscale_factors[-1],
            #                                                                   name='final_upscale')(decoder_out.tensor)
            last_output_tensor = self.op_instantiator.generate_linear_upsample(upscale_factors[-1],
                                                                               name='final_upscale')(decoder_out.tensor)
            last_output = WrappedTensor(last_output_tensor, [1, 1, filters / self.filters_ratio])
        # empty cell case
        else:
            last_output = encoder_out

        model = self._finalize_model(model_input, last_output)
        return model, self._get_output_names()

    def build_decoder_fixed_cell(self, upsample_factor: int, filters: int, decoder_input: WrappedTensor, encoder_level: WrappedTensor,
                                 decoder_index: int):
        encoder_level_filters = encoder_level.shape[-1] * self.input_shape[-1]
        encoder_features = self.op_instantiator.generate_pointwise_conv(encoder_level_filters // 2, strided=False,
                                                                        name=f'pw_conv_encoder_D{decoder_index}')(encoder_level.tensor)

        # upsampled_decoder = self.op_instantiator.generate_transpose_conv(filters, upsample_factor,
        #                                                                  name=f'tconv_D{decoder_index}')(decoder_input.tensor)
        upsampled_decoder = self.op_instantiator.generate_linear_upsample(upsample_factor,
                                                                          name=f'upsample_D{decoder_index}')(decoder_input.tensor)

        decoder_concat = layers.Concatenate(name=f'decoder_concat_D{decoder_index}')([encoder_features, upsampled_decoder])

        conv = Convolution(filters, kernel=(3, 3), strides=(1, 1), weight_reg=self.l2_weight_reg, name=f'conv_D{decoder_index}')(decoder_concat)

        return WrappedTensor(conv, elementwise_mult(decoder_input.shape, [upsample_factor] * self.op_dims + [1]))

    def _get_loss_function(self) -> losses.Loss:
        return losses.CategoricalCrossentropy()

    def _get_metrics(self) -> 'list[metrics.Metric]':
        return ['accuracy', metrics.OneHotMeanIoU(self.output_classes_count, name='mean_iou')]

    @staticmethod
    def get_results_processor_class() -> Type[BaseTrainingResults]:
        return SegmentationTrainingResults