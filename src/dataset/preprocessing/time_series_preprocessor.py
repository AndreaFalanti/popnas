from typing import Optional

import tensorflow as tf
from tensorflow.keras import layers, Sequential

from dataset.preprocessing.data_preprocessor import DataPreprocessor, AUTOTUNE


class TimeSeriesPreprocessor(DataPreprocessor):
    def __init__(self, to_one_hot: Optional[int] = None, normalize: bool = False, rescale: float = 1):
        super().__init__()
        self.to_one_hot = to_one_hot
        self.normalize = normalize
        self.rescale = rescale

    def _apply_preprocessing_on_dataset_pipeline(self, ds: tf.data.Dataset) -> tf.data.Dataset:
        # convert to one-hot encoding
        if self.to_one_hot is not None:
            ds = ds.map(lambda x, y: (x, tf.one_hot(y, self.to_one_hot)), num_parallel_calls=AUTOTUNE)

        # TODO: delete this or move to model-embeddable preprocessor?
        if self.rescale != 1:
            ds = ds.map(lambda x, y: (x / self.rescale, y))

        return ds

    def _build_preprocessing_embeddable_model(self, ds: tf.data.Dataset) -> Optional[Sequential]:
        preprocessing_keras_layers = []

        if self.normalize:
            # last axis has the features of each time step, we want to normalize each feature independently
            norm = layers.Normalization(axis=-1)
            # tested with debugger, computes the average and mean on all dataset correctly in this way
            norm.adapt(ds.map(lambda x, y: x, num_parallel_calls=AUTOTUNE))
            preprocessing_keras_layers.append(norm)

        # if at least a model-embedded layer exists, generate the Keras preprocessing model, otherwise return None
        return Sequential(layers=preprocessing_keras_layers, name='preprocessing') if len(preprocessing_keras_layers) > 0 \
            else None


