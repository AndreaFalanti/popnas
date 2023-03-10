import unittest

from dacite import from_dict

from models.generators import *
from search_space import CellSpecification, BlockSpecification
from utils.config_dataclasses import TrainingHyperparametersConfig, ArchitectureHyperparametersConfig

train_hp = from_dict(data_class=TrainingHyperparametersConfig, data={
        "epochs": 21,
        "learning_rate": 0.01,
        "weight_decay": 5e-4,
        "drop_path": 0.0,
        "softmax_dropout": 0.0,
        "optimizer": {
                "type": "adamW",
                "scheduler": "cd"
            }
    })


class TestNetworkGraphs(unittest.TestCase):
    def build_architecture_parameters(self, f: int, m: int, n: int, use_lb_reshape: bool, use_residuals: bool):
        return from_dict(data_class=ArchitectureHyperparametersConfig, data={
            "filters": f,
            "motifs": m,
            "normal_cells_per_motif": n,
            "block_join_operator": "add",
            "lookback_reshape": use_lb_reshape,
            "concat_only_unused_blocks": True,
            "residual_cells": use_residuals,
            "se_cell_output": False,
            "multi_output": False
        })

    def initialize_model_gen(self, model_gen_class: type[BaseModelGenerator], input_shape: tuple, classes: int,
                             m: int, n: int, f: int, lb_reshape: bool, residuals: bool):
        arc_params = self.build_architecture_parameters(f, m, n, lb_reshape, residuals)

        return model_gen_class(train_hp, arc_params, training_steps_per_epoch=100,
                               output_classes_count=classes, input_shape=input_shape)

    def test_simple_block_network(self):
        input_shape = (32, 32, 3)
        classes = 10
        model_gen = self.initialize_model_gen(ClassificationModelGenerator, input_shape, classes, 3, 2, 24, lb_reshape=False, residuals=False)

        cell_spec = CellSpecification([BlockSpecification(-2, '3x3 conv', -2, '3x3 conv')])
        g = model_gen.build_model_graph(cell_spec)

        self.assertEqual(g.get_total_params(), 274474, 'Single block wrong params count')

    def test_simple_block_lb_reshape_network(self):
        input_shape = (32, 32, 3)
        classes = 10
        model_gen = self.initialize_model_gen(ClassificationModelGenerator, input_shape, classes, 3, 2, 24, lb_reshape=True, residuals=False)

        cell_spec = CellSpecification([BlockSpecification(-2, '3x3 conv', -2, '3x3 conv')])
        g = model_gen.build_model_graph(cell_spec)

        self.assertEqual(g.get_total_params(), 296602, 'Lookback reshape wrong params count')

    def test_cvt_block(self):
        input_shape = (32, 32, 3)
        classes = 10
        model_gen = self.initialize_model_gen(ClassificationModelGenerator, input_shape, classes, 2, 2, 24, lb_reshape=True, residuals=False)

        cell_spec = CellSpecification([BlockSpecification(-2, '3k-1h-1b cvt', -2, '3k-1h-1b cvt')])
        g = model_gen.build_model_graph(cell_spec)
        self.assertEqual(g.get_total_params(), 155818, 'CVT wrong params count')

        cell_spec = CellSpecification([BlockSpecification(-2, '3k-1h scvt', -2, '3k-1h scvt')])
        g = model_gen.build_model_graph(cell_spec)
        self.assertEqual(g.get_total_params(), 112666, 'SCVT wrong params count')

    def test_multi_block_cell(self):
        input_shape = (64, 64, 3)
        classes = 10
        model_gen = self.initialize_model_gen(ClassificationModelGenerator, input_shape, classes, 3, 2, 24, lb_reshape=True, residuals=False)

        cell_spec = CellSpecification([BlockSpecification(-2, '2x2 maxpool', -1, '3x3 conv'),
                                       BlockSpecification(-2, '3x3 conv', -1, '3x3 conv'),
                                       BlockSpecification(0, '2x2 maxpool', 1, '2x2 maxpool'),
                                       BlockSpecification(-2, '1x7-7x1 conv', 0, '3x3 conv')])

        g = model_gen.build_model_graph(cell_spec)
        self.assertEqual(g.get_total_params(), 1667098, 'Multi-block cell wrong params count')

    def test_residual_cells(self):
        input_shape = (32, 32, 3)
        classes = 10
        model_gen = self.initialize_model_gen(ClassificationModelGenerator, input_shape, classes, 3, 2, 24, lb_reshape=False, residuals=True)

        cell_spec = CellSpecification([BlockSpecification(-2, '3x3 dconv', -2, '1x5-5x1 conv')])
        g = model_gen.build_model_graph(cell_spec)
        self.assertEqual(g.get_total_params(), 209989, 'Residual single block wrong params count')

        cell_spec = CellSpecification([BlockSpecification(-2, '2x2 maxpool', -1, '1x5-5x1 conv'),
                                       BlockSpecification(-2, '2x2 avgpool', -1, '1x5-5x1 conv'),
                                       BlockSpecification(-2, '1x1 conv', 0, '1x5-5x1 conv')])
        g = model_gen.build_model_graph(cell_spec)
        self.assertEqual(g.get_total_params(), 1156858, 'Residual multiple blocks wrong params count')

    # TODO: now these networks use transpose convolutions instead of bilinear upsample (for XLA compatibility), so the number of parameters must be updated!
    #  They are expected to fail right now, will be updated when I have the data on a new search experiment using the new macro-architecture.
    def test_segmentation_network_graphs(self):
        input_shape = (None, None, 3)
        classes = 22
        model_gen = self.initialize_model_gen(SegmentationModelGenerator, input_shape, classes, 4, 1, 28, lb_reshape=False, residuals=True)

        cell_spec = CellSpecification([BlockSpecification(-2, '1x5-5x1 conv', -2, '2x2 maxpool')])
        g = model_gen.build_model_graph(cell_spec)
        self.assertEqual(g.get_total_params(), 872702, 'Segmentation -2 lookback only wrong params count')

        cell_spec = CellSpecification([BlockSpecification(-2, '3x3 conv', -1, '5x5:4dr conv')])
        g = model_gen.build_model_graph(cell_spec)
        self.assertEqual(g.get_total_params(), 4535746, 'Segmentation mixed lookbacks wrong params count')

        cell_spec = CellSpecification([BlockSpecification(-1, '1x7-7x1 conv', -1, '8r SE'),
                                       BlockSpecification(-1, '2x2 maxpool', 0, '5x5:2dr conv'),
                                       BlockSpecification(-1, '2x2 avgpool', 0, '5x5:2dr conv')])
        g = model_gen.build_model_graph(cell_spec)
        self.assertEqual(g.get_total_params(), 10947030, 'Segmentation multi-block wrong params count')


if __name__ == '__main__':
    unittest.main()
