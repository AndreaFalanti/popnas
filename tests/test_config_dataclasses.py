import unittest

from dacite import from_dict
from mergedeep import merge

from utils.config_dataclasses import *


class TestConfigDataclasses(unittest.TestCase):
    example_config = {
        "search_space": {
            "blocks": 2,
            "lookback_depth": 2,
            "operators": [
                "identity",
                "3x3 conv"
            ]
        },
        "search_strategy": {
            "max_children": 2,
            "max_exploration_children": 1,
            "score_metric": "accuracy",
            "additional_pareto_objectives": ["time", "params"]
        },
        "training_hyperparameters": {
            "epochs": 3,
            "learning_rate": 0.01,
            "weight_reg": 5e-4,
            "drop_path_prob": 0.0,
            "softmax_dropout": 0.0,
            "optimizer": {
                "type": "adamW",
                "scheduler": "cd"
            }
        },
        "architecture_hyperparameters": {
            "filters": 24,
            "motifs": 2,
            "normal_cells_per_motif": 1,
            "block_join_operator": "add",
            "lookback_reshape": False,
            "concat_only_unused_blocks": True,
            "residual_cells": True,
            "se_cell_output": False,
            "multi_output": False
        },
        "dataset": {
            "type": "image_classification",
            "name": "cifar10",
            # "path": None,
            "classes_count": 10,
            "batch_size": 96,
            "inference_batch_size": 16,
            "validation_size": 0.1,
            "cache": True,
            "folds": 1,
            "samples": 2000,
            "balance_class_losses": False,
            # "resize": {
            #     "enabled": False,
            #     "width": 28,
            #     "height": 28
            # },
            "data_augmentation": {
                "enabled": True,
                "perform_on_gpu": False
            }
        },
        "others": {
            "accuracy_predictor_ensemble_units": 2,
            "predictions_batch_size": 512,
            "save_children_weights": False,
            "save_children_as_onnx": False,
            "pnas_mode": False,
            "train_strategy": "GPU",
            # "enable_XLA_compilation": False
        }
    }

    post_search_partial_config = {
        "training_hyperparameters": {
            "epochs": 5,
            "drop_path_prob": 0.2,
            "softmax_dropout": 0.0
        },
        "architecture_hyperparameters": {
            "multi_output": True
        },
        "dataset": {
            "folds": 1,
            "samples": 100,
            "data_augmentation": {
                "enabled": True,
                "use_cutout": True,
                "perform_on_gpu": False
            }
        }
    }
    
    def test_run_config_init(self):
        # should handle nested typing, optional fields and default value
        config = from_dict(data_class=RunConfig, data=self.example_config)

        # just do simple typing checks to see if everything is fine, the library has already lots of tests :)
        self.assertIsInstance(config, RunConfig)
        self.assertIsInstance(config.dataset, DatasetConfig)
        self.assertIsInstance(config.dataset.data_augmentation, DataAugmentationDict)

        # optional param (default: None) not provided in config
        self.assertIsNone(config.dataset.path)

        # param with default value, not provided in config
        self.assertFalse(config.others.enable_XLA_compilation)

    def test_post_search_config_merge(self):
        complete_post_search_config_dict = merge({}, self.example_config, self.post_search_partial_config)
        post_search_config_dclass = from_dict(data_class=RunConfig, data=complete_post_search_config_dict)

        self.assertEqual(post_search_config_dclass.training_hyperparameters.epochs, 5)
        self.assertTrue(post_search_config_dclass.dataset.data_augmentation.use_cutout)


if __name__ == '__main__':
    unittest.main()
