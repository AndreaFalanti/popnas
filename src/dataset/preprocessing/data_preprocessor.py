from abc import ABC, abstractmethod

import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE


class DataPreprocessor(ABC):
    '''
    Abstract class defining the interface for dataset preprocessing strategy pattern.
    '''
    def __init__(self):
        pass

    @abstractmethod
    def apply_preprocessing(self, ds: tf.data.Dataset) -> tf.data.Dataset:
        pass
