from typing import Optional

import tensorflow as tf
from tensorflow.keras import layers, Sequential

from dataset.preprocessing.data_preprocessor import DataPreprocessor, AUTOTUNE


class ImagePreprocessor(DataPreprocessor):
    def __init__(self, resize_dim: Optional['tuple[int, int]'], rescaling: Optional['tuple[float, float]'], to_one_hot: Optional[int],
                 resize_labels: bool = False):
        '''
        Preprocessor which can be applied to datasets composed of images to apply some basic transformations.

        Args:
            resize_dim: target size for image resizing.
            rescaling: transformation to apply on pixel values, as multiplier and offset, e.g. (1/255., 0) to normalize RGB in [0, 1] interval.
            to_one_hot: encode labels as one-hot. Can be used on int labels.
            resize_labels: when labels are images (e.g. masks in segmentation tasks), set it to True to resize both X and Y.
        '''
        super().__init__()
        self.resize_dim = resize_dim
        self.rescaling = rescaling
        self.to_one_hot = to_one_hot
        self.resize_labels = resize_labels

    def _apply_preprocessing_on_dataset_pipeline(self, ds: tf.data.Dataset) -> tf.data.Dataset:
        if self.resize_dim is not None:
            if self.resize_labels:
                # resize labels with nearest neighbors interpolation to avoid class artifacts
                ds = ds.map(lambda x, y: (tf.image.resize(x, self.resize_dim), tf.image.resize(y, self.resize_dim, method='nearest')),
                            num_parallel_calls=AUTOTUNE)
            else:
                ds = ds.map(lambda x, y: (tf.image.resize(x, self.resize_dim), y), num_parallel_calls=AUTOTUNE)

        # rescale pixel values (usually to [-1, 1] or [0, 1] domain), independent from dataset values so can be directly applied
        if self.rescaling is not None:
            normalization_layer = layers.Rescaling(self.rescaling[0], offset=self.rescaling[1])
            ds = ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=AUTOTUNE)

        # convert to one-hot encoding
        if self.to_one_hot is not None:
            ds = ds.map(lambda x, y: (x, tf.one_hot(y, self.to_one_hot)), num_parallel_calls=AUTOTUNE)

        return ds

    def _build_preprocessing_embeddable_model(self, ds: tf.data.Dataset) -> Optional[Sequential]:
        # no layers which require fitting on data here
        return None


