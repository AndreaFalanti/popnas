from typing import Optional

import tensorflow as tf
from tensorflow.keras import layers

from datasets.preprocessing.data_preprocessor import DataPreprocessor

AUTOTUNE = tf.data.AUTOTUNE


class ImagePreprocessor(DataPreprocessor):
    def __init__(self, resize_dim: Optional['tuple[int, int]'], rescaling: Optional['tuple[float, float]'], to_one_hot: Optional[int]):
        super().__init__()
        self.resize_dim = resize_dim
        self.rescaling = rescaling
        self.to_one_hot = to_one_hot

    def apply_preprocessing(self, ds: tf.data.Dataset):
        if self.resize_dim is not None:
            ds = ds.map(lambda x, y: (tf.image.resize(x, self.resize_dim), y), num_parallel_calls=AUTOTUNE)

        # rescale pixel values (usually to [-1, 1] or [0, 1] domain)
        if self.rescaling is not None:
            normalization_layer = layers.Rescaling(self.rescaling[0], offset=self.rescaling[1])
            ds = ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=AUTOTUNE)

        # convert to one-hot encoding
        if self.to_one_hot is not None:
            ds = ds.map(lambda x, y: (x, tf.one_hot(y, self.to_one_hot)), num_parallel_calls=AUTOTUNE)

        return ds
