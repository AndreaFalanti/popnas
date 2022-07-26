from abc import ABC, abstractmethod

import tensorflow as tf


class DataPreprocessor(ABC):
    '''
    Abstract class defining the interface for dataset preprocessing strategy pattern.
    '''
    def __init__(self):
        pass

    @abstractmethod
    def apply_preprocessing(self, ds: tf.data.Dataset):
        pass
