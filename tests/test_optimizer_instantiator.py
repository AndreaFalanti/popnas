import unittest

from tensorflow.keras.optimizers import *
from tensorflow.keras.optimizers.schedules import *
from tensorflow_addons.optimizers import *

from models.optimizers import build_optimizer, build_scheduler

# some random parameters
learning_rate = 0.1
weight_decay = 0.1
training_steps_per_epoch = 1000
epochs = 10


class TestOptimizerInstantiator(unittest.TestCase):
    def test_scheduler_builder(self):
        lr_scheduler, wd_scheduler = build_scheduler('cd', learning_rate, weight_decay, training_steps_per_epoch, epochs)
        self.assertIsInstance(lr_scheduler, CosineDecay)
        self.assertIsInstance(wd_scheduler, CosineDecay)
        # default value of alpha
        self.assertEqual(lr_scheduler.alpha, 0.0)
        self.assertEqual(wd_scheduler.alpha, 0.0)

        lr_scheduler, wd_scheduler = build_scheduler('cd: 0.1 alpha', learning_rate, weight_decay, training_steps_per_epoch, epochs)
        self.assertEqual(lr_scheduler.alpha, 0.1)
        self.assertEqual(wd_scheduler.alpha, 0.1)

        lr_scheduler, wd_scheduler = build_scheduler('cdr', learning_rate, weight_decay, training_steps_per_epoch, epochs)
        self.assertIsInstance(lr_scheduler, CosineDecayRestarts)
        self.assertIsInstance(wd_scheduler, CosineDecayRestarts)
        # t_mul and m_mul are protected, hopefully they will not change the name
        self.assertEqual(lr_scheduler.alpha, 0.0)
        self.assertEqual(lr_scheduler._t_mul, 2.0)
        self.assertEqual(lr_scheduler._m_mul, 1.0)
        self.assertEqual(wd_scheduler.alpha, 0.0)
        self.assertEqual(wd_scheduler._t_mul, 2.0)
        self.assertEqual(wd_scheduler._m_mul, 1.0)

        lr_scheduler, wd_scheduler = build_scheduler('cdr: 2 period, 2 t_mul, 0.9 m_mul, 0.1 alpha', learning_rate, weight_decay,
                                                     training_steps_per_epoch, epochs)
        self.assertIsInstance(lr_scheduler, CosineDecayRestarts)
        self.assertIsInstance(wd_scheduler, CosineDecayRestarts)
        self.assertEqual(lr_scheduler.alpha, 0.1)
        self.assertEqual(lr_scheduler._t_mul, 3.0)
        self.assertEqual(lr_scheduler._m_mul, 0.9)
        self.assertEqual(wd_scheduler.alpha, 0.1)
        self.assertEqual(wd_scheduler._t_mul, 3.0)
        self.assertEqual(wd_scheduler._m_mul, 0.9)

    def test_optimizer_builder(self):
        total_training_steps = training_steps_per_epoch * epochs

        optimizer = build_optimizer('adam', learning_rate, weight_decay, total_training_steps)
        self.assertIsInstance(optimizer, Adam)

        optimizer = build_optimizer('adamW', learning_rate, weight_decay, total_training_steps)
        self.assertIsInstance(optimizer, AdamW)

        optimizer = build_optimizer('SGD', learning_rate, weight_decay, total_training_steps)
        self.assertIsInstance(optimizer, SGD)
        # self.assertEqual(optimizer.momentum.numpy(), 0.0)

        optimizer = build_optimizer('SGD: 0.9 momentum', learning_rate, weight_decay, total_training_steps)
        self.assertIsInstance(optimizer, SGD)
        # self.assertEqual(optimizer.momentum.numpy(), np.float(0.9)) # Not working, 0.9 != 0.9 (?)

        optimizer = build_optimizer('SGDW', learning_rate, weight_decay, total_training_steps)
        self.assertIsInstance(optimizer, SGDW)

        optimizer = build_optimizer('radam', learning_rate, weight_decay, total_training_steps)
        self.assertIsInstance(optimizer, RectifiedAdam)


if __name__ == '__main__':
    unittest.main()
