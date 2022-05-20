import tensorflow as tf
import tensorflow_probability as tfp
from scipy.stats import spearmanr
from tensorflow.keras.losses import Loss, MeanSquaredError
from tensorflow.python.keras.utils import losses_utils


@tf.function
def get_rank(y_pred):
    rank = tf.argsort(y_pred)
    rank = tf.argsort(rank) + 1  # +1 to get the rank starting in 1 instead of 0
    return rank


@tf.function
def spearman_correlation_coeff(x, y):
    cov = tfp.stats.covariance(x, y, sample_axis=0, event_axis=None)
    sd_x = tfp.stats.stddev(x, sample_axis=0, keepdims=False, name=None)
    sd_y = tfp.stats.stddev(y, sample_axis=0, keepdims=False, name=None)
    # 1- because we want to minimize loss
    return 1 - cov / (sd_x * sd_y)  # + 1e-16)


@tf.function
def spearman_correlation_loss(y_true, y_pred):
    # First we obtain the ranking of the predicted values
    # y_pred_rank = tf.map_fn(lambda x: get_rank(x), y_pred, fn_output_signature=tf.int32)
    if len(y_true) == 1:
        return 0.0

    y_pred_rank = get_rank(tf.squeeze(y_pred))
    y_pred_rank = tf.cast(y_pred_rank, dtype=tf.float32)

    y_true_rank = get_rank(tf.squeeze(y_true))
    y_true_rank = tf.cast(y_true_rank, dtype=tf.float32)

    # Spearman rank correlation between each pair of samples (using map_fn), for example:
    # Sample dim: (1, 8)
    # Batch of samples dim: (batch_size, 8)
    # Output dim: (batch_size, ) with reduce mean
    #sp = tf.map_fn(lambda x: sp_rank(x[0], x[1]), (y_true, y_pred_rank), fn_output_signature=tf.float32)
    sp = spearman_correlation_coeff(y_true_rank, y_pred_rank)

    # Reduce to a single value (not actually necessary since we are in case (batch, 1) predictions)
    # loss = tf.reduce_mean(sp)
    return sp


# already implemented, but would run in eager mode. Also, it seems to not work properly.
def get_spearman_rankcor(y_true, y_pred):
    return (tf.py_function(spearmanr, [tf.cast(y_pred, tf.float32),
                                       tf.cast(y_true, tf.float32)], Tout=tf.float32))


class Spearman(Loss):
    def call(self, y_true, y_pred):
        return spearman_correlation_loss(y_true, y_pred)


class MSEWithSpearman(Loss):

    ''' Mean squared error loss, rescaled with a factor based on the spearman correlation coefficient computed on the batch. '''

    def __init__(self, spearman_weight: float = 1.0, reduction=losses_utils.ReductionV2.AUTO, name=None):
        super().__init__(reduction, name)
        self.spearman_weight = spearman_weight

    def call(self, y_true, y_pred):
        mse = MeanSquaredError().call(y_true, y_pred)
        spearman_loss = spearman_correlation_loss(y_true, y_pred)   # is in [0, 2] interval

        # multiply mse based on the spearman correlation
        # if spearman is 1, loss will be simply the plain mse
        # if spearman is -1, loss will be mse * (1 + 2 * spearman_weight), heavily penalizing the training
        # this could help the NN to correctly rank the predictions, while also minimizing the predictions error from real value
        return mse * (1 + spearman_loss * self.spearman_weight)
