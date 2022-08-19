import random

import tensorflow as tf
from tensorflow.keras.layers import Layer


# TODO: Seems to work correctly now, but it is written for easier debug and it's surely inefficient right now.
#  Investigate on better usage of TF API and rewrite it with performance in mind.
# TODO: must be adapted for different input sizes (was intended for 2D)
class ScheduledDropPath(Layer):
    '''
    This scheduled drop path implementation is inspired by the one implemented in Tensorflow for PNASNetV5 in these links:
    https://github.com/chenxi116/PNASNet.TF/blob/338371ffc3122498dc71aff9d59001f40ef22e6c/cell.py#L136
    https://github.com/tensorflow/models/blob/30e6e03f66efad4e43f1b98ec8680451f5a86a72/research/slim/nets/nasnet/nasnet_utils.py#L432.

    But these versions actually differs from the one used in FractalNet, since they don't check that at least one path survives!
    This leads to awful accuracies also on training set, since a cell/block could output a totally 0-ed tensor.
    Refactored to work on a list of tensors, before "join" layers (add, concatenate).
    '''
    def __init__(self, keep_probability: float, cell_ratio: float, total_training_steps: int, name='scheduled_drop_path', **kwargs):
        super().__init__(name=name, **kwargs)
        self.keep_probability = keep_probability
        self.cell_ratio = cell_ratio  # (self._cell_num + 1) / float(self._total_num_cells)
        self.total_training_steps = total_training_steps  # number of times weights are updated (batches_per_epoch * epochs)
        self.current_step = tf.Variable(0, trainable=False, dtype=tf.float32)

    def call(self, inputs, training=None, mask=None):
        if training and self.keep_probability < 1.0:
            # Scale keep prob by cell number
            keep_prob = 1 - self.cell_ratio * (1 - self.keep_probability)

            # Decrease keep prob over time (global_step is the current batch number)
            # current_step = tf.cast(tf.compat.v1.train.get_or_create_global_step(), tf.float32)
            current_ratio = self.current_step / self.total_training_steps
            keep_prob = 1 - current_ratio * (1 - keep_prob)

            # Drop path
            # noise_shape = [tf.shape(input=inputs)[0], 1, 1, 1]
            # random_tensor = keep_prob
            # random_tensor += tf.random.uniform(noise_shape, dtype=tf.float32)
            # binary_tensor = tf.cast(tf.floor(random_tensor), inputs.dtype)

            noise_shape = [tf.shape(input=inputs[0])[0], 1, 1, 1]
            input_dtype = inputs[0].dtype
            binary_tensors = []
            for i in range(len(inputs)):
                random_tensor = keep_prob
                random_tensor += tf.random.uniform(noise_shape, dtype=tf.float32)
                binary_tensors.append(tf.cast(tf.floor(random_tensor), dtype=input_dtype))

            agg_mask_sum = tf.math.add_n(binary_tensors)
            ensure_path_tensor = tf.maximum(0.0, 1 - agg_mask_sum)
            # mask_random_index = tf.random.uniform(shape=[], maxval=len(inputs), dtype=tf.int32)
            mask_random_index = random.randrange(len(inputs))

            binary_tensors[mask_random_index] = tf.add(binary_tensors[mask_random_index], ensure_path_tensor)

            keep_prob_inv = tf.cast(1.0 / keep_prob, dtype=input_dtype)
            self.current_step.assign_add(delta=1)

            output_tensors = []
            for i in range(len(inputs)):
                output_tensors.append(tf.multiply(tf.multiply(inputs[i], keep_prob_inv), binary_tensors[i]))

            # return inputs * keep_prob_inv * binary_tensor
            return output_tensors
        else:
            return inputs

    def get_config(self):
        config = super().get_config()
        config.update({
            'keep_probability': self.keep_probability,
            'cell_ratio': self.cell_ratio,
            'total_training_steps': self.total_training_steps
        })
        return config
