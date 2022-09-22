import random

import tensorflow as tf
from tensorflow.keras.layers import Layer


# TODO: Seems to work correctly now, but it is written for easier debug and it's surely inefficient right now.
#  Investigate on better usage of TF API and rewrite it with performance in mind.
class ScheduledDropPath(Layer):
    '''
    This scheduled drop path implementation is inspired by the one implemented in Tensorflow for PNASNetV5 in these links:
    https://github.com/chenxi116/PNASNet.TF/blob/338371ffc3122498dc71aff9d59001f40ef22e6c/cell.py#L136
    https://github.com/tensorflow/models/blob/30e6e03f66efad4e43f1b98ec8680451f5a86a72/research/slim/nets/nasnet/nasnet_utils.py#L432.

    But these versions actually differs from the one used in FractalNet, since they don't check that at least one path survives!
    This leads to awful accuracies also on training set, since a cell/block could output a totally 0-ed tensor.

    Refactored to work on a list of tensors, before "join" layers (add, concatenate).
    It scales the keep probability based on the fraction of (current_train_step / total_train_step) and (cell_index / num_cells) to apply it
    more in later stages of training and in final cells of the architecture.
    One path always survives in this implementation.
    Output values are
    '''
    def __init__(self, keep_probability: float, cell_ratio: float, total_training_steps: int, dims: int, name='scheduled_drop_path', **kwargs):
        super().__init__(name=name, **kwargs)
        self.keep_probability = keep_probability
        # cell ratio is in (0, 1] interval. It is =1 for the last cell, while previous cells scale for their index.
        # (self._cell_num + 1) / float(self._total_num_cells)
        self.cell_ratio = cell_ratio
        # number of times weights are updated (batches_per_epoch * epochs)
        self.total_training_steps = total_training_steps
        # keep track of current execution step
        self.current_step = tf.Variable(0, trainable=False, dtype=tf.float32)
        # input tensors ndims, used to generate the masks
        self.dims = dims

    def call(self, inputs, training=None, mask=None):
        if training and self.keep_probability < 1.0:
            # Scale keep prob by cell number
            keep_prob = 1 - self.cell_ratio * (1 - self.keep_probability)

            # Decrease keep prob over time (drop more in later stages)
            current_ratio = self.current_step / self.total_training_steps
            keep_prob = 1 - current_ratio * (1 - keep_prob)

            # make a noise shape with same dimensionality of input.
            # First dim is equal to the batch size, others are equal to 1 (single value for each batch sample).
            noise_shape = [tf.shape(input=inputs[0])[0]] + [1] * self.dims
            input_dtype = inputs[0].dtype

            # A mask for each input tensors, if =0 then the path is dropped
            binary_tensors = []
            for i in range(len(inputs)):
                random_tensor = keep_prob
                random_tensor += tf.random.uniform(noise_shape, dtype=tf.float32)
                binary_tensors.append(tf.cast(tf.floor(random_tensor), dtype=input_dtype))

            # make sure at least one path survives (agg_mask_sum will be =0 if all paths should be dropped, so maximum force it to 1)
            # ensure_path_tensor is summed to a random mask (random index), making it survive
            agg_mask_sum = tf.math.add_n(binary_tensors)
            ensure_path_tensor = tf.maximum(0.0, 1 - agg_mask_sum)
            mask_random_index = random.randrange(len(inputs))
            binary_tensors[mask_random_index] = tf.add(binary_tensors[mask_random_index], ensure_path_tensor)

            keep_prob_inv = tf.cast(1.0 / keep_prob, dtype=input_dtype)
            # increment current_step, keep track of how much training steps have been executed to scale the probability correctly
            self.current_step.assign_add(delta=1)

            # multiply the output tensors for their masks
            output_tensors = []
            for i in range(len(inputs)):
                output_tensors.append(tf.multiply(tf.multiply(inputs[i], keep_prob_inv), binary_tensors[i]))

            # old return value, now is done in for loop above but kept for reference
            # follows the behavior of dropout layer
            # return inputs * keep_prob_inv * binary_tensor
            return output_tensors
        else:
            return inputs

    def get_config(self):
        config = super().get_config()
        config.update({
            'keep_probability': self.keep_probability,
            'cell_ratio': self.cell_ratio,
            'total_training_steps': self.total_training_steps,
            'dims': self.dims
        })
        return config
