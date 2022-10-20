from abc import ABC, abstractmethod
from typing import Optional

import tensorflow as tf
from tensorflow.keras import Sequential

AUTOTUNE = tf.data.AUTOTUNE


class DataPreprocessor(ABC):
    '''
    Abstract class defining the interface for dataset preprocessing strategy pattern.
    '''
    def __init__(self):
        self.preprocessor_model = None

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

