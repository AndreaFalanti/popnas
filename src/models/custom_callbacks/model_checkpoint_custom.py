from tensorflow.keras import callbacks


class ModelCheckpointCustom(callbacks.ModelCheckpoint):
    '''
    Same of Keras ModelCheckpoint, but can save a single checkpoint between *save_chunk* epochs easily.
    At most 1 checkpoint is saved every *save_chunk* epochs, only the best one in the interval is stored since it overrides the others
    (e.g. if save_chunk=10 and epoch 124 is best among [120, 129] -> e124 weights saved in cp_ec12_10.ckpt).

    More effective and easier to use than fiddling with *save_freq*, since it is not strictly periodic, but still avoids saving too many checkpoints.
    '''
    def __init__(self, filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', save_freq='epoch',
                 options=None, save_chunk: int = 10, **kwargs):
        super().__init__(filepath, monitor, verbose, save_best_only, save_weights_only, mode, save_freq, options, **kwargs)

        self.save_chunk = save_chunk

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch // self.save_chunk, logs)
