import importlib
from logging import Logger
from typing import Union

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import datasets, layers, Sequential
from tensorflow.keras.utils import to_categorical

AUTOTUNE = tf.data.AUTOTUNE


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
    # TODO: untested legacy code, not sure this is working
    else:
        spec = importlib.util.spec_from_file_location(dataset_source)
        dataset_source = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dataset_source)
        (x_train, y_train), (x_test, y_test) = dataset_source.load_data()
        classes_count = None

    return (x_train, y_train), (x_test, y_test), classes_count


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


def __build_tf_datasets(samples_fold: 'tuple[list, list, list, list]', batch_size: int, data_augmentation: Sequential, augment_on_gpu: bool):
    '''
    Build the training and validation datasets to be used in model.fit().
    '''

    x_train, y_train, x_val, y_val = samples_fold

    # create a batched dataset, cached in memory for better performance
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size).cache()
    validation_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size).cache().prefetch(AUTOTUNE)

    # if data augmentation is performed on CPU, map it before prefetch, otherwise just prefetch
    train_dataset = train_dataset.prefetch(AUTOTUNE) if augment_on_gpu \
        else train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    return train_dataset, validation_dataset


def generate_tensorflow_datasets(dataset_config: dict, logger: Logger):
    '''
    Generates training and validation tensorflow datasets for each fold to perform, based on the provided configuration parameters.
    Args:
        dataset_config:
        logger:

    Returns:
        list of dataset tuples (one for each fold), the number of classes, training batch count and validation batch count
    '''
    dataset_name = dataset_config['name']
    dataset_path = dataset_config['path']
    dataset_folds_count = dataset_config['folds'] if dataset_config['folds'] > 0 else 1
    samples_limit = dataset_config['samples']
    dataset_classes_count = dataset_config['classes_count']
    batch_size = dataset_config['batch_size']
    val_size = dataset_config['validation_size']

    data_augmentation_config = dataset_config['data_augmentation']
    use_data_augmentation = data_augmentation_config['enabled']
    augment_on_gpu = data_augmentation_config['perform_on_gpu']

    dataset_folds = []  # type: list[tuple[tf.data.Dataset, tf.data.Dataset]]

    train, test, classes = __load_dataset_images(dataset_name)
    dataset_classes_count = classes or dataset_classes_count    # like || in javascript
    # TODO: produce also the test dataset
    (x_train_init, y_train_init), _ = __preprocess_images(train, test, dataset_classes_count, samples_limit)

    if use_data_augmentation:
        # follow similar augmentation techniques used in other papers, which usually are:
        # - horizontal flip
        # - 4px translate on both height and width [fill=reflect] (sometimes upscale to 40x40, with random crop to original 32x32)
        # - whitening (not always used)
        data_augmentation = Sequential([
            layers.experimental.preprocessing.RandomFlip('horizontal'),
            # layers.experimental.preprocessing.RandomRotation(20/360),   # 20 degrees range
            # layers.experimental.preprocessing.RandomZoom(height_factor=0.1, width_factor=0.1),
            layers.experimental.preprocessing.RandomTranslation(height_factor=0.125, width_factor=0.125)
        ], name='data_augmentation')
    else:
        data_augmentation = None

    # TODO: is it ok to generate the splits by shuffling randomly?
    for i in range(dataset_folds_count):
        logger.info('Preprocessing and building dataset fold #%d...', i + 1)

        # create a validation set for evaluation of the child models
        x_train, x_validation, y_train, y_validation = train_test_split(x_train_init, y_train_init, test_size=val_size, stratify=y_train_init)

        dataset_folds.append(__build_tf_datasets((x_train, y_train, x_validation, y_validation), batch_size, data_augmentation, augment_on_gpu))

    train_batches = int(np.ceil(len(x_train) / batch_size))
    val_batches = int(np.ceil(len(x_validation) / batch_size))

    return dataset_folds, classes, train_batches, val_batches
