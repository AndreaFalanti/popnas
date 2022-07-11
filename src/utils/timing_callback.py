from timeit import default_timer as timer

import tensorflow as tf


class TimingCallback(tf.keras.callbacks.Callback):
    def __init__(self, logs={}):
        self.logs = []

    def on_train_batch_begin(self, epoch, logs={}):
        self.start_time = timer()

    def on_train_batch_end(self, epoch, logs={}):
        self.logs.append(timer() - self.start_time)


class InferenceTimingCallback(tf.keras.callbacks.Callback):
    def __init__(self, logs={}):
        self.logs = []

    def on_predict_batch_begin(self, batch, logs={}):
        self.start_time = timer()

    def on_predict_batch_end(self, batch, logs={}):
        self.logs.append(timer() - self.start_time)
