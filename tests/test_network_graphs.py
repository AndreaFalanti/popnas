import unittest

from models.generators import ClassificationModelGenerator
from search_space import CellSpecification, BlockSpecification


class TestNetworkGraphs(unittest.TestCase):
    def build_cnn_hp(self, f):
        return {
            "epochs": 21,
            "learning_rate": 0.01,
            "filters": f,
            "weight_reg": 5e-4,
            "use_adamW": True,
            "drop_path_prob": 0.0,
            "cosine_decay_restart": {
                "enabled": True,
                "period_in_epochs": 3,
                "t_mul": 2.0,
                "m_mul": 1.0,
                "alpha": 0.0
            },
            "softmax_dropout": 0.0
        }

    def build_architecture_parameters(self, m, n, use_lb_reshape):
        return {
            "motifs": m,
            "normal_cells_per_motif": n,
            "block_join_operator": "add",
            "lookback_reshape": use_lb_reshape,
            "concat_only_unused_blocks": True,
            "residual_cells": False,
            "se_cell_output": False,
            "multi_output": False
        }

    def initialize_model_gen(self, input_shape: tuple, classes: int, m: int, n: int, f: int, lb_reshape: bool):
        cnn_hp = self.build_cnn_hp(f)
        arc_params = self.build_architecture_parameters(m, n, lb_reshape)

        return ClassificationModelGenerator(cnn_hp, arc_params, training_steps_per_epoch=100,
                                            output_classes_count=classes, input_shape=input_shape)

    def test_simple_block_network(self):
        input_shape = (32, 32, 3)
        classes = 10
        model_gen = self.initialize_model_gen(input_shape, classes, 3, 2, 24, lb_reshape=False)

        cell_spec = CellSpecification([BlockSpecification(-2, '3x3 conv', -2, '3x3 conv')])
        g = model_gen.build_model_graph(cell_spec)

        self.assertEqual(g.get_total_params(), 274474, 'Single block wrong params count')

    def test_simple_block_lb_reshape_network(self):
        input_shape = (32, 32, 3)
        classes = 10
        model_gen = self.initialize_model_gen(input_shape, classes, 3, 2, 24, lb_reshape=True)

        cell_spec = CellSpecification([BlockSpecification(-2, '3x3 conv', -2, '3x3 conv')])
        g = model_gen.build_model_graph(cell_spec)

        self.assertEqual(g.get_total_params(), 296602, 'Lookback reshape wrong params count')

    def test_cvt_block(self):
        input_shape = (32, 32, 3)
        classes = 10
        model_gen = self.initialize_model_gen(input_shape, classes, 2, 2, 24, lb_reshape=True)

        cell_spec = CellSpecification([BlockSpecification(-2, '3k-1h-1b cvt', -2, '3k-1h-1b cvt')])
        g = model_gen.build_model_graph(cell_spec)
        self.assertEqual(g.get_total_params(), 155818, 'CVT wrong params count')

        cell_spec = CellSpecification([BlockSpecification(-2, '3k-1h scvt', -2, '3k-1h scvt')])
        g = model_gen.build_model_graph(cell_spec)
        self.assertEqual(g.get_total_params(), 112666, 'SCVT wrong params count')

    def test_multi_block_cell(self):
        input_shape = (64, 64, 3)
        classes = 10
        model_gen = self.initialize_model_gen(input_shape, classes, 3, 2, 24, lb_reshape=True)

        cell_spec = CellSpecification([BlockSpecification(-2, '2x2 maxpool', -1, '3x3 conv'),
                                       BlockSpecification(-2, '3x3 conv', -1, '3x3 conv'),
                                       BlockSpecification(0, '2x2 maxpool', 1, '2x2 maxpool'),
                                       BlockSpecification(-2, '1x7-7x1 conv', 0, '3x3 conv')])

        g = model_gen.build_model_graph(cell_spec)
        self.assertEqual(g.get_total_params(), 1667098, 'Multi-block cell wrong params count')


if __name__ == '__main__':
    unittest.main()
