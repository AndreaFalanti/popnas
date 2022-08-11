from abc import ABC, abstractmethod
from typing import Optional

import tensorflow as tf
from tensorflow.keras import Sequential

import log_service
from dataset.preprocessing import DataPreprocessor

AUTOTUNE = tf.data.AUTOTUNE


class BaseDatasetGenerator(ABC):
    '''
    Abstract class defining the interface for dataset generator factory pattern.
    The right subclass should be instantiated from the task type given in configuration (e.g. image classification).
    '''
    def __init__(self, dataset_config: dict):
        self._logger = log_service.get_logger(__name__)

        self.dataset_name = dataset_config['name']  # type: str
        self.dataset_path = dataset_config['path']
        self.dataset_folds_count = dataset_config['folds'] if dataset_config['folds'] > 0 else 1
        self.samples_limit = dataset_config['samples']
        self.dataset_classes_count = dataset_config['classes_count']
        self.batch_size = dataset_config['batch_size']
        self.val_size = dataset_config['validation_size']
        self.cache = dataset_config['cache']

        data_augmentation_config = dataset_config['data_augmentation']
        self.use_data_augmentation = data_augmentation_config['enabled']
        self.augment_on_gpu = data_augmentation_config['perform_on_gpu']

    def _finalize_dataset(self, ds: tf.data.Dataset, batch_size: Optional[int], preprocessor: DataPreprocessor,
                           data_augmentation: Optional[Sequential],
                           shard_policy: tf.data.experimental.AutoShardPolicy = tf.data.experimental.AutoShardPolicy.DATA):
        '''
        Complete the dataset pipelines with the operations common to all different implementations (keras, tfds, and custom loaded with keras).
        Basically apply batch, preprocessing, cache, data augmentation (only to training set) and prefetch.

        Returns:
            dataset, batches count
        '''
        # set sharding options to DATA. This improves performances on distributed environments.
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = shard_policy
        ds = ds.with_options(options)

        # create a batched dataset (if batch is provided, otherwise is assumed to be already batched)
        if batch_size is not None:
            ds = ds.batch(batch_size)

        # PREPROCESSING
        ds = preprocessor.apply_preprocessing(ds)

        # after preprocessing, cache in memory for better performance, if enabled. Should be disabled only for large datasets.
        if self.cache:
            ds = ds.cache()

        # if data augmentation is performed on CPU, map it before prefetch
        if data_augmentation is not None and not self.augment_on_gpu:
            ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)

        return ds.prefetch(AUTOTUNE), len(ds)

    @abstractmethod
    def generate_train_val_datasets(self) -> 'tuple[list[tf.data.Dataset, tf.data.Dataset], int, tuple, int, int]':
        '''
            Generates training and validation tensorflow datasets for each fold to perform, based on the provided configuration parameters.

            Returns:
                list of dataset tuples (one for each fold), the number of classes, the input shape, training batch count and validation batch count
            '''
        raise NotImplementedError()

    @abstractmethod
    def generate_test_dataset(self) -> 'tuple[tf.data.Dataset, int, tuple, int]':
        '''
        Generates test tensorflow datasets, based on the provided configuration parameters.

        Returns:
            test dataset, the number of classes, the input shape, test batch count
        '''
        raise NotImplementedError()