import math
import os.path
from collections import Counter
from logging import Logger
from typing import Union

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
from tensorflow.keras import datasets, layers, Sequential
from tensorflow.keras.utils import to_categorical, image_dataset_from_directory

AUTOTUNE = tf.data.AUTOTUNE


def __finalize_datasets(train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, batch_size: int,
                        data_augmentation: Sequential, augment_on_gpu: bool, cache: bool):
    '''
    Complete the dataset pipelines with the operations common to all different implementations (keras, tfds, and custom loaded with keras).
    Basically apply batch, cache, data augmentation (only to training set) and prefetch.

    Returns:
        train dataset, validation dataset, train batches count, validation batches count
    '''

    # create a batched dataset
    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)

    # cache in memory for better performance, if enabled
    if cache:
        train_ds = train_ds.cache()
        val_ds = val_ds.cache()

    # if data augmentation is performed on CPU, map it before prefetch
    if data_augmentation is not None and not augment_on_gpu:
        train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)

    train_batches = len(train_ds)
    val_batches = len(val_ds)

    return train_ds.prefetch(AUTOTUNE), val_ds.prefetch(AUTOTUNE), train_batches, val_batches


def __load_dataset_images(dataset_source: str):
    '''
    Load images of a dataset
    Args:
        dataset_source: A valid Keras dataset name or a path to a dataset location

    Returns:
        train images, test images and the classes contained in the dataset
    '''
    if dataset_source == 'cifar10':
        (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
        classes_count = 10
    elif dataset_source == 'cifar100':
        (x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()
        classes_count = 100
    elif dataset_source == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()
        classes_count = 10
        # add dimension since the images are in grayscale (dimension 1 is omitted)
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)
    else:
        raise ValueError('unsupported dataset name')

    image_shape = x_train.shape[1:]

    return (x_train, y_train), (x_test, y_test), classes_count, image_shape


def __preprocess_images(train: tuple, test: tuple, classes_count: int, samples_limit: Union[int, None]):
    x_train, y_train = train
    x_test, y_test = test

    if samples_limit is not None:
        x_train = x_train[:samples_limit]
        y_train = y_train[:samples_limit]

    # normalize images into [0,1] domain
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    # transform labels to one-hot encoding, so that categorical crossentropy can be used
    y_train = to_categorical(y_train, classes_count)
    y_test = to_categorical(y_test, classes_count)

    return (x_train, y_train), (x_test, y_test)


def __build_tf_datasets(samples_fold: 'tuple[list, list, list, list]'):
    '''
    Build the training and validation datasets to be used in model.fit().
    '''

    x_train, y_train, x_val, y_val = samples_fold

    # create the tf datasets from numpy arrays
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    validation_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))

    return train_dataset, validation_dataset


def __generate_datasets_from_tfds(dataset_name: str, samples_limit: Union[int, None], validation_size: float, resize_dim: 'tuple[int, int]'):
    split_spec = 'train' if samples_limit is None else f'train[:{samples_limit}]'
    train_ds, info = tfds.load(dataset_name, split=split_spec, as_supervised=True, shuffle_files=True, with_info=True,
                               read_config=tfds.ReadConfig(try_autocache=False))  # type: tf.data.Dataset, tfds.core.DatasetInfo

    # convert to one-hot encoding
    classes = info.features['label'].num_classes
    train_ds = train_ds.map(lambda x, y: (x, tf.one_hot(y, classes)), num_parallel_calls=AUTOTUNE)

    if validation_size is None:
        val_split_spec = 'validation' if samples_limit is None else f'validation[:{samples_limit}]'
        val_ds = tfds.load(dataset_name, split=val_split_spec, as_supervised=True, shuffle_files=True,
                           read_config=tfds.ReadConfig(try_autocache=False))  # type: tf.data.Dataset
        val_ds = val_ds.map(lambda x, y: (x, tf.one_hot(y, classes)), num_parallel_calls=AUTOTUNE)
    else:
        samples_count = samples_limit or info.splits['train'].num_examples
        train_samples = math.ceil(samples_count * (1 - validation_size))

        val_ds = train_ds.skip(train_samples)
        train_ds = train_ds.take(train_samples)

    if resize_dim is not None:
        train_ds = train_ds.map(lambda x, y: (tf.image.resize(x, resize_dim), y), num_parallel_calls=AUTOTUNE)
        val_ds = val_ds.map(lambda x, y: (tf.image.resize(x, resize_dim), y), num_parallel_calls=AUTOTUNE)

    return train_ds, val_ds, info


def generate_tensorflow_datasets(dataset_config: dict, logger: Logger):
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
    resize_dim = (resize_config['width'], resize_config['height']) if resize_config['enabled'] else None

    data_augmentation_config = dataset_config['data_augmentation']
    use_data_augmentation = data_augmentation_config['enabled']
    augment_on_gpu = data_augmentation_config['perform_on_gpu']

    dataset_folds = []  # type: list[tuple[tf.data.Dataset, tf.data.Dataset]]
    data_augmentation = get_data_augmentation_model() if use_data_augmentation else None

    # Custom dataset, loaded with Keras
    if dataset_path is not None:
        if resize_dim is None:
            raise ValueError('Image must have a set resize dimension to use a custom dataset')

        if val_size is None:
            train_ds = image_dataset_from_directory(os.path.join(dataset_path, 'keras_training'), label_mode='categorical',
                                                    image_size=resize_dim, batch_size=batch_size)
            val_ds = image_dataset_from_directory(os.path.join(dataset_path, 'keras_validation'), label_mode='categorical',
                                                  image_size=resize_dim, batch_size=batch_size)
        # extract a validation split from training samples
        else:
            train_ds = image_dataset_from_directory(os.path.join(dataset_path, 'keras_training'), validation_split=val_size, seed=123,
                                                    subset='training', label_mode='categorical', image_size=resize_dim, batch_size=batch_size)
            val_ds = image_dataset_from_directory(os.path.join(dataset_path, 'keras_training'), validation_split=val_size, seed=123,
                                                  subset='validation', label_mode='categorical', image_size=resize_dim, batch_size=batch_size)

        # debug
        # train_labels_perc = get_dataset_stratification(train_ds)
        # val_labels_perc = get_dataset_stratification(val_ds)
        # print('Train labels distribution: ' + str(train_labels_perc))
        # print('Validation labels distribution: ' + str(val_labels_perc))

        # normalize into [0, 1] domain
        normalization_layer = tf.keras.layers.Rescaling(1. / 255)
        train_ds = train_ds.map(lambda x, y: (normalization_layer(x, training=True), y), num_parallel_calls=AUTOTUNE)
        val_ds = val_ds.map(lambda x, y: (normalization_layer(x, training=True), y), num_parallel_calls=AUTOTUNE)

        # cache in memory for better performance, if enabled
        if cache:
            train_ds = train_ds.cache()
            val_ds = val_ds.cache()

        # if data augmentation is performed on CPU, map it before prefetch
        if data_augmentation is not None and not augment_on_gpu:
            train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)

        train_ds = train_ds.prefetch(AUTOTUNE)
        val_ds = val_ds.prefetch(AUTOTUNE)

        train_batches = len(train_ds)
        val_batches = len(val_ds)

        dataset_folds.append((train_ds, val_ds))

        classes = dataset_classes_count
        image_shape = resize_dim + (3,)
    # Keras dataset case
    elif dataset_name in ['cifar10', 'cifar100', 'fashion_mnist']:
        train, test, classes, image_shape = __load_dataset_images(dataset_name)
        dataset_classes_count = classes or dataset_classes_count    # like || in javascript
        (x_train_init, y_train_init), _ = __preprocess_images(train, test, dataset_classes_count, samples_limit)

        for i in range(dataset_folds_count):
            logger.info('Preprocessing and building dataset fold #%d...', i + 1)

            # create a validation set for evaluation of the child models
            x_train, x_val, y_train, y_val = train_test_split(x_train_init, y_train_init, test_size=val_size, stratify=y_train_init)
            train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
            val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
            train_ds, val_ds, train_batches, val_batches = __finalize_datasets(train_ds, val_ds, batch_size, data_augmentation, augment_on_gpu, cache)

            dataset_folds.append((train_ds, val_ds))
    # TFDS case
    else:
        train_ds, val_ds, info = __generate_datasets_from_tfds(dataset_name, samples_limit, val_size, resize_dim)
        train_ds, val_ds, train_batches, val_batches = __finalize_datasets(train_ds, val_ds, batch_size, data_augmentation, augment_on_gpu, cache)

        image_shape = info.features['image'].shape if resize_dim is None else resize_dim + (info.features['image'].shape[2],)
        classes = info.features['label'].num_classes

        dataset_folds.append((train_ds, val_ds))

    logger.info('Dataset folds built successfully')

    return dataset_folds, classes, image_shape, train_batches, val_batches


def get_data_augmentation_model():
    '''
    Keras model that can be used in both CPU or GPU for data augmentation.
    Follow similar augmentation techniques used in other papers, which usually are:
    - horizontal flip
    - 4px translate on both height and width [fill=reflect] (sometimes upscale to 40x40, with random crop to original 32x32)
    - whitening (not always used, here it's not performed)
    '''
    return Sequential([
        layers.experimental.preprocessing.RandomFlip('horizontal'),
        # layers.experimental.preprocessing.RandomRotation(20/360),   # 20 degrees range
        # layers.experimental.preprocessing.RandomZoom(height_factor=0.1, width_factor=0.1),
        layers.experimental.preprocessing.RandomTranslation(height_factor=0.125, width_factor=0.125)
    ], name='data_augmentation')


def get_dataset_labels_distribution(ds: tf.data.Dataset) -> 'dict[int, float]':
    ''' Returns the percentage of samples associated to each label of the dataset. '''
    label_counter = Counter()

    for (x, y) in ds:
        label_counter.update(np.argmax(y, axis=-1))

    percentages_dict = {}
    for (k, v) in label_counter.items():
        percentages_dict[k] = v / sum(label_counter.values())

    return percentages_dict


def generate_balanced_weights_for_classes(ds: tf.data.Dataset) -> 'dict[int, float]':
    ''' Compute weights to balance the importance during training of classes with unbalanced amounts of samples. '''
    class_percentages_dict = get_dataset_labels_distribution(ds)
    num_classes = len(class_percentages_dict.keys())

    class_weights_dict = {}
    for cl, perc in class_percentages_dict.items():
        class_weights_dict[cl] = 1 / (num_classes * perc)

    return class_weights_dict
