import tensorflow as tf
from einops import rearrange
from tensorflow.keras import losses
from tensorflow_ranking.python.utils import sorted_ranks


class SquaredRankError(losses.Loss):
    '''
    Compute the squared rank error between the predictions and the true labels.
    It is normalized between [0, 1) by dividing for (num_elements)^2.
    '''
    def call(self, y_true, y_pred):
        # from (batch, 1) to (1, batch), since the ranks are computed on the last dimension
        blist_y_true = rearrange(y_true, 'b p -> p b')
        blist_y_pred = rearrange(y_pred, 'b p -> p b')

        # convert to ranks
        true_scores = tf.cast(sorted_ranks(blist_y_true), dtype='float32')
        pred_scores = tf.cast(sorted_ranks(blist_y_pred), dtype='float32')

        # the max rank diff is actually the number of elements - 1, but if only one element is present, then we would get a NaN loss :(
        max_rank_diff = tf.cast(tf.shape(true_scores)[-1], dtype='float32')
        sre = tf.math.squared_difference(true_scores, pred_scores) / tf.square(max_rank_diff)

        return tf.squeeze(sre)


class MSEWithSRE(losses.Loss):
    '''
    Mean squared error loss plus squared rank error (SRE) loss, rescaled by an arbitrary factor.
    SRE loss is useful to improve the ranking of the predictions.
    '''
    def __init__(self, sre_weight: float = 1.0, reduction=losses.Reduction.AUTO, name=None):
        super().__init__(reduction, name)
        self.sre_weight = sre_weight
        self.mse_loss = losses.MeanSquaredError()
        self.sre_loss = SquaredRankError()

    def call(self, y_true, y_pred):
        mse = self.mse_loss.call(y_true, y_pred)
        sre_loss = self.sre_loss.call(y_true, y_pred)

        # combine the two losses
        return mse + self.sre_weight * sre_loss
