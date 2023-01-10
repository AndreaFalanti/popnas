from timeit import default_timer as timer

from tensorflow.keras.callbacks import Callback


class InferenceTimingCallback(Callback):
    '''
    Custom callback used to record the inference time required for each batch.
    Make sure it is the first callback to avoid measuring execution time due to other callbacks.
    '''
    def __init__(self):
        super().__init__()

        self.logs = []
        self.start_time = 0

    def on_predict_batch_begin(self, batch, logs=None):
        self.start_time = timer()

    def on_predict_batch_end(self, batch, logs=None):
        self.logs.append(timer() - self.start_time)
