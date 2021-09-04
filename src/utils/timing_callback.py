import tensorflow as tf
from timeit import default_timer as timer

class TimingCallback(tf.keras.callbacks.Callback):
    def __init__(self, logs={}):
        self.logs=[]
    def on_epoch_begin(self, epoch, logs={}):
        self.start_time = timer()
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer()-self.start_time)
    