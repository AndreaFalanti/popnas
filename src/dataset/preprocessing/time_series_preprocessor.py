from typing import Optional

import tensorflow as tf

from dataset.preprocessing.data_preprocessor import DataPreprocessor, AUTOTUNE


class TimeSeriesPreprocessor(DataPreprocessor):
    def __init__(self, to_one_hot: Optional[int] = None):
        super().__init__()
        self.to_one_hot = to_one_hot

    def apply_preprocessing(self, ds: tf.data.Dataset) -> tf.data.Dataset:
        # convert to one-hot encoding
        if self.to_one_hot is not None:
            ds = ds.map(lambda x, y: (x, tf.one_hot(y, self.to_one_hot)), num_parallel_calls=AUTOTUNE)

        return ds
