from typing import Optional

import tensorflow as tf
from tensorflow.keras import layers

from dataset.preprocessing.data_preprocessor import DataPreprocessor, AUTOTUNE


class TimeSeriesPreprocessor(DataPreprocessor):
    def __init__(self, to_one_hot: Optional[int] = None, normalize: bool = False, rescale: float = 1):
        super().__init__()
        self.to_one_hot = to_one_hot
        self.normalize = normalize
        self.rescale = rescale

    def apply_preprocessing(self, ds: tf.data.Dataset) -> tf.data.Dataset:
        # convert to one-hot encoding
        if self.to_one_hot is not None:
            ds = ds.map(lambda x, y: (x, tf.one_hot(y, self.to_one_hot)), num_parallel_calls=AUTOTUNE)

        if self.rescale != 1:
            ds = ds.map(lambda x, y: (x / self.rescale, y))

        if self.normalize:
            # last axis is features of the time step, we want to normalize each feature independently for the entire timestamp
            norm = layers.Normalization(axis=-1)
            norm.adapt(ds.map(lambda x, y: x))
            ds = ds.map(lambda x, y: (norm(x), y))

        return ds
