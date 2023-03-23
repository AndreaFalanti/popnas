import tensorflow as tf
from einops import rearrange
from tensorflow.keras import losses
from tensorflow_ranking.python.keras import losses as tfr_losses


class PairwiseHingeFromPred(losses.Loss):
    '''
    Compute the pairwise Hinge loss directly from prediction values (automatically converts y_true and y_pred to scores compared to TF ranking).
    '''

    def __init__(self, reduction=losses.Reduction.AUTO, name=None):
        super().__init__(reduction, name)
        self.pw_hinge_loss = tfr_losses.PairwiseHingeLoss()

    def call(self, y_true, y_pred):
        # from (batch, 1) to (1, batch), since the ranks are computed on the last dimension
        blist_y_true = rearrange(y_true, 'b p -> p b')
        blist_y_pred = rearrange(y_pred, 'b p -> p b')

        # true_scores = tf.cast(sorted_ranks(blist_y_true), dtype='float32')
        # pred_scores = tf.cast(sorted_ranks(blist_y_pred), dtype='float32')

        # loss is in shape (1, batch), make it (batch,) so that can be applied correctly
        loss = self.pw_hinge_loss.call(blist_y_true, blist_y_pred)
        return tf.squeeze(loss)


class MSEWithPairwiseHinge(losses.Loss):
    '''
    Mean squared error loss plus pairwise Hinge loss, rescaled by an arbitrary factor.
    Pairwise Hinge loss is useful to improve the ranking of the predictions.
    '''
    def __init__(self, hinge_weight: float = 0.1, reduction=losses.Reduction.AUTO, name=None):
        super().__init__(reduction, name)
        self.hinge_weight = hinge_weight
        self.mse_loss = losses.MeanSquaredError()
        self.pw_hinge_loss = PairwiseHingeFromPred()

    def call(self, y_true, y_pred):
        mse = self.mse_loss.call(y_true, y_pred)
        hinge_loss = self.pw_hinge_loss.call(y_true, y_pred)

        # combine the two losses
        return mse + self.hinge_weight * hinge_loss
