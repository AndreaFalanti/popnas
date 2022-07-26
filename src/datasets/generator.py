import math
import os.path
from logging import Logger
from typing import Union, Optional

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
from tensorflow.keras import datasets, Sequential
from tensorflow.keras.utils import image_dataset_from_directory

from datasets.augmentation import get_image_data_augmentation_model
from datasets.preprocessing import ImagePreprocessor, DataPreprocessor

AUTOTUNE = tf.data.AUTOTUNE


def __finalize_dataset(ds: tf.data.Dataset, batch_size: Optional[int], preprocessor: DataPreprocessor,
                       data_augmentation: Optional[Sequential], augment_on_gpu: bool, cache: bool,
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
    if cache:
        ds = ds.cache()

    # if data augmentation is performed on CPU, map it before prefetch
    if data_augmentation is not None and not augment_on_gpu:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)

    return ds.prefetch(AUTOTUNE), len(ds)


def __load_keras_dataset_images(dataset_source: str):
    '''
    Load images of a Keras dataset. In this case the data is in form of Numpy arrays.
    Args:
        dataset_source: A valid Keras dataset name or a path to a dataset location

    Returns:
        train images, test images and the classes contained in the dataset
    '''
    if dataset_source == 'cifar10':
        (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
        classes_count = 10
        channels = 3
    elif dataset_source == 'cifar100':
        (x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()
        classes_count = 100
        channels = 3
    elif dataset_source == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()
        classes_count = 10
        channels = 1
        # add dimension since the images are in grayscale (dimension 1 is omitted)
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)
    else:
        raise ValueError('unsupported dataset name')

    image_shape = x_train.shape[1:]

    return (x_train, y_train), (x_test, y_test), classes_count, image_shape, channels


def __generate_datasets_from_tfds(dataset_name: str, samples_limit: Union[int, None], validation_size: float):
    split_spec = 'train' if samples_limit is None else f'train[:{samples_limit}]'
    train_ds, info = tfds.load(dataset_name, split=split_spec, as_supervised=True, shuffle_files=True, with_info=True,
                               read_config=tfds.ReadConfig(try_autocache=False))  # type: tf.data.Dataset, tfds.core.DatasetInfo

    if validation_size is None:
        val_split_spec = 'validation' if samples_limit is None else f'validation[:{samples_limit}]'
        val_ds = tfds.load(dataset_name, split=val_split_spec, as_supervised=True, shuffle_files=True,
                           read_config=tfds.ReadConfig(try_autocache=False))  # type: tf.data.Dataset
    else:
        samples_count = samples_limit or info.splits['train'].num_examples
        train_samples = math.ceil(samples_count * (1 - validation_size))

        val_ds = train_ds.skip(train_samples)
        train_ds = train_ds.take(train_samples)

    return train_ds, val_ds, info


def generate_train_val_datasets(dataset_config: dict, logger: Logger):
    '''
    Generates training and validation tensorflow datasets for each fold to perform, based on the provided configuration parameters.

    Args:
        dataset_config:
        logger:

    Returns:
        list of dataset tuples (one for each fold), the number of classes, the image shape, training batch count and validation batch count
    '''
    dataset_name = dataset_config['name']   # type: str
    dataset_path = dataset_config['path']
    dataset_folds_count = dataset_config['folds'] if dataset_config['folds'] > 0 else 1
    samples_limit = dataset_config['samples']
    dataset_classes_count = dataset_config['classes_count']
    batch_size = dataset_config['batch_size']
    val_size = dataset_config['validation_size']
    cache = dataset_config['cache']

    resize_config = dataset_config['resize']
    resize_dim = (resize_config['height'], resize_config['width']) if resize_config['enabled'] else None

    data_augmentation_config = dataset_config['data_augmentation']
    use_data_augmentation = data_augmentation_config['enabled']
    augment_on_gpu = data_augmentation_config['perform_on_gpu']

    dataset_folds = []  # type: list[tuple[tf.data.Dataset, tf.data.Dataset]]
    data_augmentation = get_image_data_augmentation_model() if use_data_augmentation else None

    # TODO: folds were implemented only in Keras datasets, later i have externalized the logic to support the other dataset format. Still,
    #  the train-validation is done in different ways based on format right now, and if seed is fixed the folds could be identical.
    #  Folds right now are never used, if this feature become relevant check that folds are actually stochastic, or use K-fold et simila.
    for i in range(dataset_folds_count):
        logger.info('Preprocessing and building dataset fold #%d...', i + 1)
        shard_policy = tf.data.experimental.AutoShardPolicy.DATA

        # Custom dataset, loaded with Keras utilities
        if dataset_path is not None:
            if resize_dim is None:
                raise ValueError('Image must have a set resize dimension to use a custom dataset')

            # TODO: samples limit is applied but in a naive way, since unbatching breakes the dataset cardinality.
            #  Stratification is done due stochasticity, which is not good for low amount of samples.
            #  Anyway not a big problem since in "real" runs you would not limit the samples deliberately.
            # TODO: new tf versions make batching optional, making possible to batch the dataset in finalize like the others.
            if val_size is None:
                train_ds = image_dataset_from_directory(os.path.join(dataset_path, 'keras_training'), label_mode='categorical',
                                                        image_size=resize_dim, batch_size=batch_size)
                val_ds = image_dataset_from_directory(os.path.join(dataset_path, 'keras_validation'), label_mode='categorical',
                                                      image_size=resize_dim, batch_size=batch_size)

                # limit only train dataset, if limit is set
                if samples_limit is not None:
                    logger.info('Limiting training dataset to %d batches', samples_limit // batch_size)
                    train_ds = train_ds.take(samples_limit // batch_size)
            # extract a validation split from training samples
            else:
                # TODO: find a way to make the split stratified (Stratification is done due stochasticity right now)
                train_ds = image_dataset_from_directory(os.path.join(dataset_path, 'keras_training'), validation_split=val_size, seed=123,
                                                        subset='training', label_mode='categorical', image_size=resize_dim, batch_size=batch_size)
                val_ds = image_dataset_from_directory(os.path.join(dataset_path, 'keras_training'), validation_split=val_size, seed=123,
                                                      subset='validation', label_mode='categorical', image_size=resize_dim, batch_size=batch_size)

                # limit both datasets, so that the sum of samples is the declared one
                if samples_limit is not None:
                    training_samples = math.ceil(samples_limit * (1 - val_size) // batch_size)
                    val_samples = math.floor(samples_limit * val_size // batch_size)

                    logger.info('Limiting training dataset to %d batches', training_samples)
                    train_ds = train_ds.take(training_samples)
                    logger.info('Limiting validation dataset to %d batches', val_samples)
                    val_ds = val_ds.take(val_samples)

            # TODO: sharding should be set to FILE here? or not?
            shard_policy = tf.data.experimental.AutoShardPolicy.FILE

            classes = dataset_classes_count
            image_shape = resize_dim + (3,)
        # Keras dataset case
        elif dataset_name in ['cifar10', 'cifar100', 'fashion_mnist']:
            train, test, classes, image_shape, channels = __load_keras_dataset_images(dataset_name)
            classes = classes or dataset_classes_count    # like || in javascript

            x_train_init, y_train_init = train
            if samples_limit is not None:
                x_train_init = x_train_init[:samples_limit]
                y_train_init = y_train_init[:samples_limit]

            if resize_dim is not None:
                image_shape = resize_dim + (channels,)

            # create a validation set for evaluation of the child models
            x_train, x_val, y_train, y_val = train_test_split(x_train_init, y_train_init, test_size=val_size, stratify=y_train_init)
            # remove last axis
            y_train = np.squeeze(y_train)
            y_val = np.squeeze(y_val)

            train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
            val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        # TFDS case
        else:
            train_ds, val_ds, info = __generate_datasets_from_tfds(dataset_name, samples_limit, val_size)

            image_shape = info.features['image'].shape if resize_dim is None else resize_dim + (info.features['image'].shape[2],)
            classes = info.features['label'].num_classes

        # finalize dataset generation, common logic to all dataset formats
        # avoid categorical for custom datasets, since already done
        using_custom_ds = dataset_path is not None
        data_preprocessor = ImagePreprocessor(resize_dim, rescaling=(1. / 255, 0), to_one_hot=None if using_custom_ds else classes)
        # TODO: remove when updating TF to new version, since custom dataset is not forced to be batched
        batch_len = None if using_custom_ds else batch_size
        train_ds, train_batches = __finalize_dataset(train_ds, batch_len, data_preprocessor, data_augmentation, augment_on_gpu,
                                                     cache, shard_policy=shard_policy)
        val_ds, val_batches = __finalize_dataset(val_ds, batch_len, data_preprocessor, None, augment_on_gpu,
                                                 cache, shard_policy=shard_policy)
        dataset_folds.append((train_ds, val_ds))

    logger.info('Dataset folds built successfully')

    # IDE is wrong, variables are always assigned since folds > 1, so at least one cycle is always executed
    return dataset_folds, classes, image_shape, train_batches, val_batches


def generate_test_dataset(dataset_config: dict, logger: Logger):
    '''
    Generates test tensorflow datasets, based on the provided configuration parameters.

    Args:
        dataset_config:
        logger:

    Returns:
        list of dataset tuples (one for each fold), the number of classes, the image shape, training batch count and validation batch count
    '''
    dataset_name = dataset_config['name']   # type: str
    dataset_path = dataset_config['path']
    dataset_classes_count = dataset_config['classes_count']
    batch_size = dataset_config['batch_size']
    cache = dataset_config['cache']

    resize_config = dataset_config['resize']
    resize_dim = (resize_config['height'], resize_config['width']) if resize_config['enabled'] else None

    shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    # Custom dataset, loaded with Keras utilities
    if dataset_path is not None:
        if resize_dim is None:
            raise ValueError('Image must have a set resize dimension to use a custom dataset')

        # TODO: new tf versions make batching optional, making possible to batch the dataset in finalize like the others.
        test_ds = image_dataset_from_directory(os.path.join(dataset_path, 'keras_test'), label_mode='categorical',
                                                image_size=resize_dim, batch_size=batch_size)

        # TODO: sharding should be set to FILE here? or not?
        shard_policy = tf.data.experimental.AutoShardPolicy.FILE

        classes = dataset_classes_count
        image_shape = resize_dim + (3,)
    # Keras dataset case
    elif dataset_name in ['cifar10', 'cifar100', 'fashion_mnist']:
        train, test, classes, image_shape, channels = __load_keras_dataset_images(dataset_name)
        classes = classes or dataset_classes_count    # like || in javascript

        x_test, y_test = test

        if resize_dim is not None:
            image_shape = resize_dim + (channels,)

        # remove last axis
        y_test = np.squeeze(y_test)

        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    # TFDS case
    else:
        test_ds, info = tfds.load(dataset_name, split='test', as_supervised=True, shuffle_files=False, with_info=True,
                                   read_config=tfds.ReadConfig(try_autocache=False))  # type: tf.data.Dataset, tfds.core.DatasetInfo

        image_shape = info.features['image'].shape if resize_dim is None else resize_dim + (info.features['image'].shape[2],)
        classes = info.features['label'].num_classes

    # finalize dataset generation, common logic to all dataset formats
    # avoid categorical for custom datasets, since already done
    using_custom_ds = dataset_path is not None
    data_preprocessor = ImagePreprocessor(resize_dim, rescaling=(1. / 255, 0), to_one_hot=None if using_custom_ds else classes)
    # TODO: remove when updating TF to new version, since custom dataset is not forced to be batched
    batch_len = None if using_custom_ds else batch_size
    test_ds, batches = __finalize_dataset(test_ds, batch_len, data_preprocessor, None, False,
                                                 cache, shard_policy=shard_policy)

    logger.info('Test dataset built successfully')
    return test_ds, classes, image_shape, batches
