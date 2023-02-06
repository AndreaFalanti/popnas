import tensorflow as tf
from tensorflow.keras import layers, Sequential


def keras_normalization():
    # last axis has the features of each time step, we want to normalize each feature independently
    norm = layers.Normalization(axis=-1)

    def fit_on_dataset(ds: tf.data.Dataset):
        # tested with debugger, computes the average and mean on all dataset correctly in this way
        norm.adapt(ds.map(lambda x, y: x, num_parallel_calls=tf.data.AUTOTUNE))
        return Sequential(layers=[norm], name='preprocessing')

    return fit_on_dataset
