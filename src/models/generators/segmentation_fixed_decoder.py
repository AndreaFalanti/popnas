from typing import Type, Optional

import tensorflow as tf
from keras import Sequential
from tensorflow.keras import layers, metrics, Model, losses

from models.generators.base import BaseModelGenerator, NetworkBuildInfo, WrappedTensor
from models.graphs.network_graph import NetworkGraph
from models.graphs.units import TensorNode
from models.operators.layers import AtrousSpatialPyramidPooling, Convolution, PoolingConv, Pooling
from models.operators.params_utils import compute_aspp_params, compute_conv_params
from models.results import BaseTrainingResults, SegmentationTrainingResults
from search_space_units import CellSpecification
from utils.config_dataclasses import TrainingHyperparametersConfig, ArchitectureHyperparametersConfig
from utils.func_utils import elementwise_mult


class SegmentationFixedDecoderModelGenerator(BaseModelGenerator):
    '''
    Model generator for image semantic segmentation tasks.
    The generated models have a DeepLab like structure,
    where each level of the encoder structure is composed of cells found during the NAS process, while the decoder is fixed.

    NOTE: these models use bilinear upsample, which is not compilable with XLA (see: https://github.com/tensorflow/tensorflow/issues/57575).
    Do not set the XLA compilation flag in config, otherwise it will cause an error during model building!
    '''

    def __init__(self, train_hp: TrainingHyperparametersConfig, arc_hp: ArchitectureHyperparametersConfig, training_steps_per_epoch: int,
                 output_classes_count: int, input_shape: 'tuple[int, ...]', data_augmentation_model: Optional[Sequential] = None,
                 preprocessing_model: Optional[Sequential] = None, save_weights: bool = False, ignore_class: Optional[int] = None):
        super().__init__(train_hp, arc_hp, training_steps_per_epoch, output_classes_count, input_shape, data_augmentation_model,
                         preprocessing_model, save_weights)

        self.ignore_class = ignore_class

    def _generate_network_info(self, cell_spec: CellSpecification, use_stem: bool) -> NetworkBuildInfo:
        blocks = len(cell_spec)
        use_skip = -2 in cell_spec.used_lookbacks
        total_cells = self.get_maximum_cells()

        # additional info regarding the cell stack
        nearest_lookback = max(cell_spec.used_lookbacks, default=1)
        # "nearest_lookback" is negative, so the slice correctly starts from the last cell (which is the one connected to the output)
        used_cell_indexes = list(range(total_cells))[::nearest_lookback]

        reduction_cell_indexes = list(range(self.normal_cells_per_motif, total_cells, self.normal_cells_per_motif + 1))
        need_input_norm_indexes = [el - min(cell_spec.used_lookbacks) - 1 for el in reduction_cell_indexes] if use_skip and self.lookback_reshape \
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

    def _compute_decoder_upsample_factors(self):
        '''
        Inspired by DeepLab, upsample progressively with steps of factor 4.
        When using an odd number of reductions, the last upsample will have a 2x upsample.

        Returns:
            list with a number of elements equal to the needed decoder units + last upsample, indicating their upsample factor
        '''
        reductions = self.motifs - 1
        upscale_factors = [4] * (reductions // 2)
        # use 2x as last upsample when using an odd number of reductions
        if reductions % 2 == 1:
            upscale_factors.append(2)

        return upscale_factors

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
        if upscale_ratio > 1:
            h_tensor = self.op_instantiator.generate_linear_upsample(int(upscale_ratio),
                                                                     name=f'output_upsample{name_suffix}')(h_tensor)

        # TODO: a spatial dropout could be more indicated for this structure
        if dropout_prob > 0.0:
            h_tensor = layers.Dropout(dropout_prob)(h_tensor)

        output_name = self.get_output_layers_name()
        logits = layers.Conv2D(self.output_classes_count, kernel_size=(1, 1), kernel_initializer='he_uniform',
                               kernel_regularizer=self.l2_weight_reg, name=f'logits{name_suffix}')(h_tensor)
        output = layers.Activation('softmax', dtype='float32', name=f'{output_name}{name_suffix}')(logits)

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

        # Add the fixed part after last cell: ASPP, decoder and output
        final_cell_out = net.get_graph_output_node()
        final_filters = int(final_cell_out.shape[-1])

        # ASPP
        v_attributes = {
            'name': ['ASPP'],
            'op': ['ASPP'],
            'cell_index': [-1],
            'block_index': [-1],
            'params': [compute_aspp_params(final_filters, 0.5)]
        }
        net.g.add_vertices(1, v_attributes)

        edges = [(final_cell_out.name, 'ASPP')]
        edge_attributes = {
            'tensor_shape': [str(final_cell_out.shape)]
        }
        net.g.add_edges(edges, edge_attributes)

        # DECODER
        upscale_factors = self._compute_decoder_upsample_factors()
        current_node = TensorNode('ASPP', final_cell_out.shape)
        for i, upsample_factor in enumerate(upscale_factors[:-1]):
            encoder_target_node = encoder_concat_nodes[-2 * (i + 1)]
            encoder_pw_shape = elementwise_mult(encoder_target_node.shape, [1, 1, 0.5])
            output_shape = elementwise_mult(final_cell_out.shape, [4, 4, 1])
            concat_shape = elementwise_mult(final_cell_out.shape, [4, 4, 1.125])

            decoder_unit_suffix = f'_D{i}'
            v_attributes = {
                'name': [f'upsample{decoder_unit_suffix}', f'pw_encoder{decoder_unit_suffix}',
                         f'encoder_concat{decoder_unit_suffix}', f'conv{decoder_unit_suffix}'],
                'op': ['upsample', '1x1 conv', 'concat', '3x3 conv'],
                'cell_index': [-1] * 4,
                'block_index': [-1] * 4,
                'params': [0, self.op_instantiator.get_op_params('1x1 conv', encoder_target_node.shape, encoder_pw_shape),
                           0, self.op_instantiator.get_op_params('3x3 conv', concat_shape, output_shape)]
            }
            net.g.add_vertices(4, v_attributes)

            edges = [(current_node.name, f'upsample{decoder_unit_suffix}'),
                     (encoder_target_node.name, f'pw_encoder{decoder_unit_suffix}'),
                     (f'upsample{decoder_unit_suffix}', f'encoder_concat{decoder_unit_suffix}'),
                     (f'pw_encoder{decoder_unit_suffix}', f'encoder_concat{decoder_unit_suffix}'),
                     (f'encoder_concat{decoder_unit_suffix}', f'conv{decoder_unit_suffix}')]
            edge_attributes = {
                'tensor_shape': [str(final_cell_out.shape), str(encoder_pw_shape), str(output_shape),
                                 str(encoder_target_node.shape), str(concat_shape)]
            }
            net.g.add_edges(edges, edge_attributes)

            current_node = TensorNode(f'conv{decoder_unit_suffix}', output_shape)

        # OUTPUT
        v_attributes = {
            'name': ['out_pointwise_conv'],
            'op': ['1x1 conv'],
            'cell_index': [-1],
            'block_index': [-1],
            'params': [compute_conv_params((1, 1), final_filters, self.output_classes_count, bn=False)]
        }
        net.g.add_vertices(1, v_attributes)

        # add edges from inputs to operator layers, plus from operator layers to add
        edges = [(final_cell_out.name, 'out_pointwise_conv')]
        edge_attributes = {
            'tensor_shape': [str(current_node.shape)]
        }
        net.g.add_edges(edges, edge_attributes)

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

        # build encoder in empty cell case, to downsample the tensor before using ASPP and perform it on the same dimensionality
        # this step is important to closely get the training time required by elements shared by all architectures
        if M == 0:
            cell_inputs, level_outputs, filters = self._build_empty_cell_fake_encoder(cell_inputs)

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
        aspp = AtrousSpatialPyramidPooling(filters, dilation_rates=(3, 6, 12), filters_ratio=0.5,
                                           weight_reg=self.l2_weight_reg, activation_f=self.activation_f)(cell_inputs[-1].tensor)
        encoder_out = WrappedTensor(aspp, cell_inputs[-1].shape)

        # DECODER part (fixed)
        upscale_factors = self._compute_decoder_upsample_factors()

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

        # fix multi-output considering only cell outputs, so it would discard the fixed part at the end of the network (ASPP and decoder)
        if self.multi_output:
            # override last cell output with the one considering also ASPP and decoder
            self.cell_index = self.cell_index - 1
            self._generate_output(last_output, self.dropout_prob)

        model = self._finalize_model(model_input, last_output)
        return model, self._get_output_names()

    def _build_empty_cell_fake_encoder(self, cell_inputs: 'list[WrappedTensor]'):
        '''
        Build encoder for the empty cell case, downsampling the tensor before using the ASPP module.

        In this way, the ASPP is performed on the same dimensionality of other networks;
        this step is important to closely get the training time required by elements shared by all architectures.
        '''
        filters = self.filters
        decoder_needed_level_indexes = range(self.motifs - 1, 0, -2)
        cur_shape = [1, 1, self.filters_ratio]
        fake_encoder_units = []
        t = cell_inputs[-1]

        for i in range(self.motifs):
            if i in decoder_needed_level_indexes:
                pool_factor = 2 if i == 1 else 4
                filters = filters * pool_factor
                # avg pooling + pointwise, quick operation to reshape the tensor
                pool_size = pool_stride = tuple([pool_factor] * self.op_dims)
                downsampled_tensor = PoolingConv(Pooling('avg', pool_size, pool_stride, name=f'empty_cell_fake_encoder_unit_{i}'),
                                                 filters, weight_reg=self.l2_weight_reg)(t.tensor)
                cur_shape = elementwise_mult(cur_shape, [1 / pool_factor, 1 / pool_factor, pool_factor])
                t = WrappedTensor(downsampled_tensor, cur_shape)
                fake_encoder_units.append(t)
            else:
                fake_encoder_units.append(None)

        return [fake_encoder_units[-1]], fake_encoder_units[:-1], filters

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
        return losses.SparseCategoricalCrossentropy(ignore_class=self.ignore_class)

    def _get_metrics(self) -> 'list[metrics.Metric]':
        return ['accuracy', metrics.MeanIoU(self.output_classes_count, ignore_class=self.ignore_class, sparse_y_pred=False, name='mean_iou')]

    @staticmethod
    def get_results_processor_class() -> Type[BaseTrainingResults]:
        return SegmentationTrainingResults
