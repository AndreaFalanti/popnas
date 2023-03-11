from typing import Optional

import tensorflow as tf
from tensorflow.keras import layers, Sequential

from dataset.preprocessing.data_preprocessor import DataPreprocessor, AUTOTUNE


class ImagePreprocessor(DataPreprocessor):
    def __init__(self, resize_dim: Optional['tuple[int, int]'], rescaling: Optional['tuple[float, float]'], to_one_hot: Optional[int],
                 resize_labels: bool = False, pad_to_multiples_of: Optional[int] = None):
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
        self.pad_to_multiples_of = pad_to_multiples_of

    def _apply_preprocessing_on_dataset_pipeline(self, ds: tf.data.Dataset) -> tf.data.Dataset:
        if self.resize_dim is not None:
            if self.resize_labels:
                # resize labels with nearest neighbors interpolation to avoid class artifacts
                ds = ds.map(lambda x, y: (tf.image.resize(x, self.resize_dim), tf.image.resize(y, self.resize_dim, method='nearest')),
                            num_parallel_calls=AUTOTUNE)
            else:
                ds = ds.map(lambda x, y: (tf.image.resize(x, self.resize_dim), y), num_parallel_calls=AUTOTUNE)
        else:
            if self.pad_to_multiples_of is not None:
                @tf.function
                def zero_pad_to_multiples(image: tf.Tensor, label: tf.Tensor):
                    ''' Zero pad spatial dimensions of images (and masks) to be multiples of *self.pad_to_multiples_of*. '''
                    shape = tf.shape(image)
                    # shape = image.shape.as_list()
                    # get spatial dims (HW), avoid first dim if the dataset is batched (should be 4 dims)
                    h, w = shape[0], shape[1]  # if len(shape) == 3 else shape[1:3]

                    remainder_y = h % self.pad_to_multiples_of
                    remainder_x = w % self.pad_to_multiples_of
                    padding_y = 0 if remainder_y == 0 else self.pad_to_multiples_of - remainder_y
                    padding_x = 0 if remainder_x == 0 else self.pad_to_multiples_of - remainder_x

                    pads_y = [tf.math.ceil(padding_y / 2), tf.math.floor(padding_y / 2)]
                    pads_x = [tf.math.ceil(padding_x / 2), tf.math.floor(padding_x / 2)]
                    padding = [pads_y, pads_x, [0, 0]]

                    image = tf.pad(image, padding)
                    label = tf.pad(label, padding) if self.resize_labels else label
                    return image, label

                ds = ds.map(zero_pad_to_multiples, num_parallel_calls=AUTOTUNE)

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


