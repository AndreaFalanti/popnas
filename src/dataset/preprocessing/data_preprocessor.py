from abc import ABC, abstractmethod
from typing import Optional

import tensorflow as tf
from tensorflow.keras import Sequential

AUTOTUNE = tf.data.AUTOTUNE


class DataPreprocessor(ABC):
    '''
    Abstract class defining the interface for dataset preprocessing strategy pattern.
    '''
    def __init__(self, to_one_hot: Optional[int], remap_classes: 'Optional[dict[str, int]]'):
        self.preprocessor_model = None
        self.to_one_hot = to_one_hot
        self.remap_classes_dict = remap_classes

    def _apply_labels_preprocessing(self, ds: tf.data.Dataset):
        if self.remap_classes_dict is not None:
            # cast labels to int32, since lookup does not work on uint8
            ds = ds.map(lambda x, y: (x, tf.cast(y, dtype=tf.int32)), num_parallel_calls=AUTOTUNE)

            lookup_init = tf.lookup.KeyValueTensorInitializer(list(map(int, self.remap_classes_dict.keys())), list(self.remap_classes_dict.values()),
                                                              key_dtype=tf.int32, value_dtype=tf.int32)
            labels_lookup_table = tf.lookup.StaticHashTable(lookup_init, default_value=255)

            # remap labels and cast back to uint8
            ds = ds.map(lambda x, y: (x, tf.cast(labels_lookup_table.lookup(y), dtype=tf.uint8)), num_parallel_calls=AUTOTUNE)

        # convert labels to one-hot encoding
        if self.to_one_hot is not None:
            ds = ds.map(lambda x, y: (x, tf.one_hot(y, self.to_one_hot, dtype=tf.uint8)), num_parallel_calls=AUTOTUNE)

        return ds

    def apply_preprocessing(self, ds: tf.data.Dataset, fit_data: bool) -> tf.data.Dataset:
        '''
        Apply the preprocessing pipeline to the input dataset.
        Note that some operations could not be applied here, but instead must be retrieved as an additional Keras model using
        "build_preprocessing_embeddable_model" function.

        Args:
            ds: dataset to preprocess
            fit_data: should be true only when preprocessing training data, to fit potential layers like normalization which are parametrized on
                data values

        Returns:
            the preprocessed dataset
        '''
        ds = self._apply_preprocessing_on_dataset_pipeline(ds)
        ds = self._apply_labels_preprocessing(ds)
        if fit_data:
            self.preprocessor_model = self._build_preprocessing_embeddable_model(ds)

        return ds

    @abstractmethod
    def _apply_preprocessing_on_dataset_pipeline(self, ds: tf.data.Dataset) -> tf.data.Dataset:
        '''
        Apply the preprocessing to the input dataset pipeline.

        Args:
            ds: dataset to preprocess

        Returns:
            the preprocessed dataset
        '''
        pass

    @abstractmethod
    def _build_preprocessing_embeddable_model(self, ds: tf.data.Dataset) -> Optional[Sequential]:
        '''
        Layers like normalization that must be fit on training data and kept with same parameters also at inference time are not directly applied
        to the data, but instead are set up into a Keras model that must be connected after the input layer in final architectures.
        Pass this model to ModelGenerator class.

        Should be called only when generating the training split, since preprocessing layers are fit only on training data.

        In this way, preprocessing parameters become portable through the checkpoints, but as a downside the preprocessing is done
        on the training device (GPU/TPU), which can lead to small delays (CPU preprocessing is instead asynchronous).

        Args:
            ds: dataset to preprocess

        Returns:
            Keras model for techniques that must be applied directly in the final architectures, if present, otherwise None is returned
        '''
        pass

