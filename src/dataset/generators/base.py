from abc import ABC, abstractmethod
from typing import Optional, Callable, NamedTuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential

import log_service
from dataset.preprocessing import DataPreprocessor

AUTOTUNE = tf.data.AUTOTUNE
AutoShardPolicy = tf.data.experimental.AutoShardPolicy
SEED = 1234


class DatasetsFold(NamedTuple):
    ''' Container for a fold of datasets used during training (training and validation datasets). '''
    train: tf.data.Dataset
    validation: tf.data.Dataset


def load_npz(file_path: str, possibly_ragged: bool = False) -> 'tuple[np.ndarray, np.ndarray]':
    ''' Load an .npz numpy file from disk. Set possibly_ragged flag to True if the array contains arrays of different dimensions. '''
    npz = np.load(file_path, allow_pickle=possibly_ragged)
    return npz['x'], npz['y']


def generate_tf_dataset_from_numpy_ragged_array(x_arr: np.ndarray, y_arr: np.ndarray, dtype: Optional[np.dtype.type] = None):
    '''
    Generates a TF dataset from arrays of different dimensions (ragged).
    The original arrays should have dtype = object, since numpy does not support directly ragged arrays.

    Args:
        x_arr: samples in numpy array object type
        y_arr: labels in numpy array object type
        dtype: optional dtype if explicit conversion is needed

    Returns:
        a TF dataset containing ragged tensors
    '''
    x_ragged = tf.ragged.stack([np.asarray(x, dtype=dtype) for x in x_arr])
    y_ragged = tf.ragged.stack([np.asarray(y, dtype=dtype) for y in y_arr])

    # to_tensor is required to use map functions in the pipeline. It's strange, maybe ragged tensors are not necessary?
    return tf.data.Dataset.from_tensor_slices((x_ragged, y_ragged)).map(lambda x, y: (x.to_tensor(), y.to_tensor()))


def generate_possibly_ragged_dataset(x: np.ndarray, y: np.ndarray) -> tf.data.Dataset:
    '''
    If the array type is *object*, generate a ragged dataset, otherwise generate a normal one.
    A ragged dataset can contain samples of different dimensions.
    '''
    if x.dtype == np.object:
        return generate_tf_dataset_from_numpy_ragged_array(x, y, dtype=None)
    else:
        return tf.data.Dataset.from_tensor_slices((x, y))


class BaseDatasetGenerator(ABC):
    '''
    Abstract class defining the interface for dataset generator factory pattern.
    The right subclass should be instantiated from the task type given in configuration (e.g. image classification).
    '''

    def __init__(self, dataset_config: dict, enable_tpu_tricks: bool = False):
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

        self.enable_tpu_tricks = enable_tpu_tricks

    def _finalize_dataset(self, ds: tf.data.Dataset, batch_size: Optional[int], preprocessor: DataPreprocessor,
                          keras_data_augmentation: Optional[Sequential] = None, tf_data_augmentation: 'Optional[Callable]' = None,
                          shuffle: bool = False, fit_preprocessing_layers: bool = False,
                          shard_policy: AutoShardPolicy = AutoShardPolicy.DATA) -> 'tuple[tf.data.Dataset, int]':
        '''
        Complete the dataset pipelines with the operations common to all different implementations (keras, tfds, and custom loaded with keras).
        Basically apply batch, preprocessing, cache, data augmentation (only to training set) and prefetch.

        Args:
            ds: dataset to finalize
            batch_size: desired samples per batch
            preprocessor: DataPreprocessor which should be applied
            keras_data_augmentation: Keras model with data augmentation layers
            tf_data_augmentation: TF function mappable on the dataset to perform additional data augmentation features not included in Keras
            shuffle: if True, shuffle dataset before batching and also shuffle batches at each training iteration
            shard_policy: AutoShardPolicy for distributing the dataset in multi-device environments

        Returns:
            finalized dataset, batches count and the eventual preprocessing Keras model to apply in the architectures
        '''
        # set sharding options to DATA. This improves performances on distributed environments.
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = shard_policy
        ds = ds.with_options(options)

        # shuffle samples to avoid batches of same class (if samples are ordered)
        if shuffle:
            ds = ds.shuffle(len(ds), seed=SEED)

        # PREPROCESSING
        # TODO: should be done after batching, to exploit vectorization. Right now there is a problem with ragged tensors, so moved before batching.
        ds = preprocessor.apply_preprocessing(ds, fit_preprocessing_layers)

        # create a batched dataset (if batch is provided, otherwise is assumed to be already batched)
        if batch_size is not None:
            # make all batches of the same size, avoids drop remainder to not loss data, instead duplicates some samples
            if self.enable_tpu_tricks:
                duplicated_samples_count = batch_size - (len(ds) % batch_size)
                duplicate_ds = ds.take(duplicated_samples_count)
                ds = ds.concatenate(duplicate_ds)

        # after preprocessing, cache in memory for better performance, if enabled. Should be disabled only for large datasets.
        if self.cache:
            ds = ds.cache()

        # TF augmentations could not work on batches, perform them before the batching
        if tf_data_augmentation is not None:
            ds = ds.map(tf_data_augmentation, num_parallel_calls=AUTOTUNE)

        # TODO: should be better to batch before applying preprocessing/augmentation to improve performance by vectorizing the operations,
        #  but not all operators can be applied on batches.
        if batch_size is not None:
            ds = ds.batch(batch_size, num_parallel_calls=AUTOTUNE)

        # if data augmentation is performed on CPU, map it before prefetch
        # keras augmentations should always work on batches
        if keras_data_augmentation is not None and not self.augment_on_gpu:
            ds = ds.map(lambda x, y: (keras_data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)

        # shuffle batches at each epoch
        if shuffle:
            ds = ds.shuffle(len(ds), reshuffle_each_iteration=True, seed=SEED)

        return ds.prefetch(AUTOTUNE), len(ds)

    @abstractmethod
    def generate_train_val_datasets(self) -> 'tuple[list[DatasetsFold], int, tuple[int, ...], int, int, Optional[Sequential]]':
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

    @abstractmethod
    def generate_final_training_dataset(self) -> 'tuple[tf.data.Dataset, int, tuple[int, ...], int, Optional[Sequential]]':
        '''
        Generate a full training (or union of training and validation if split a priori) tensorflow dataset, used in training to convergence
        done for best model selected after the search procedure.

        If val_size is set to None, the function expect to find a separate validation set, which will be merged to
        the training one.

        Sample limit is not applied here.

        Returns:
            train dataset, the number of classes, the input shape, train batch count
        '''
        raise NotImplementedError()
