import importlib
import math
from logging import Logger
from typing import Union

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
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
    elif dataset_source == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()
        classes_count = 10
        # add dimension since the images are in grayscale (dimension 1 is omitted)
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)
    # TODO: untested legacy code, not sure this is working
    else:
        spec = importlib.util.spec_from_file_location(dataset_source)
        dataset_source = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dataset_source)
        (x_train, y_train), (x_test, y_test) = dataset_source.load_data()
        classes_count = None

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


def __generate_datasets_from_tfds(dataset_name: str, samples_limit: Union[int, None], batch_size: int, validation_size: float,
                                  data_augmentation: Sequential, augment_on_gpu: bool):
    split_spec = 'train' if samples_limit is None else f'train[:{samples_limit}]'
    train_ds, info = tfds.load(dataset_name, split=split_spec, as_supervised=True, shuffle_files=True,
                               with_info=True)  # type: tf.data.Dataset, tfds.core.DatasetInfo

    samples_count = samples_limit or info.splits['train'].num_examples
    train_samples = math.ceil(samples_count * (1 - validation_size))

    classes = info.features._feature_dict['label'].num_classes
    train_ds = train_ds.map(lambda x, y: (x, tf.one_hot(y, classes)), num_parallel_calls=AUTOTUNE)

    val_ds = train_ds.skip(train_samples).batch(batch_size).cache().prefetch(AUTOTUNE)
    train_ds = train_ds.take(train_samples).batch(batch_size).cache()

    # if data augmentation is performed on CPU, map it before prefetch, otherwise just prefetch
    train_dataset = train_ds.prefetch(AUTOTUNE) if augment_on_gpu \
        else train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    return train_dataset, val_ds, info


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

    data_augmentation_config = dataset_config['data_augmentation']
    use_data_augmentation = data_augmentation_config['enabled']
    augment_on_gpu = data_augmentation_config['perform_on_gpu']

    dataset_folds = []  # type: list[tuple[tf.data.Dataset, tf.data.Dataset]]
    data_augmentation = get_data_augmentation_model() if use_data_augmentation else None

    if dataset_name in ['cifar10', 'cifar100', 'fashion_mnist']:
        train, test, classes, image_shape = __load_dataset_images(dataset_name)
        dataset_classes_count = classes or dataset_classes_count    # like || in javascript
        # TODO: produce also the test dataset
        (x_train_init, y_train_init), _ = __preprocess_images(train, test, dataset_classes_count, samples_limit)

        # TODO: is it ok to generate the splits by shuffling randomly?
        for i in range(dataset_folds_count):
            logger.info('Preprocessing and building dataset fold #%d...', i + 1)

            # create a validation set for evaluation of the child models
            x_train, x_validation, y_train, y_validation = train_test_split(x_train_init, y_train_init, test_size=val_size, stratify=y_train_init)

            dataset_folds.append(__build_tf_datasets((x_train, y_train, x_validation, y_validation), batch_size, data_augmentation, augment_on_gpu))

        train_batches = int(np.ceil(len(x_train) / batch_size))
        val_batches = int(np.ceil(len(x_validation) / batch_size))
    # TODO: separate version for fast development, but it's similar to the generic dataset function. Integrate this part when possible.
    # used only in final tests
    elif dataset_name.startswith('imagenet2012'):
        train_ds, info = tfds.load(dataset_name, split='train', as_supervised=True, shuffle_files=True,
                                   with_info=True)  # type: tf.data.Dataset, tfds.core.DatasetInfo
        val_ds, info = tfds.load(dataset_name, split='validation', as_supervised=True, shuffle_files=True,
                                 with_info=True)  # type: tf.data.Dataset, tfds.core.DatasetInfo

        classes = info.features._feature_dict['label'].num_classes
        train_ds = train_ds.map(lambda x, y: (x, tf.one_hot(y, classes)), num_parallel_calls=AUTOTUNE)

        # resize images
        resize_dim = (160, 160)
        train_ds = train_ds.map(lambda x, y: (tf.image.resize(x, resize_dim), y))
        val_ds = val_ds.map(lambda x, y: (tf.image.resize(x, resize_dim), y))

        val_ds = val_ds.batch(batch_size).cache().prefetch(AUTOTUNE)
        train_ds = train_ds.batch(batch_size).cache()

        # if data augmentation is performed on CPU, map it before prefetch, otherwise just prefetch
        train_ds = train_ds.prefetch(AUTOTUNE) if augment_on_gpu \
            else train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

        image_shape = resize_dim + (3,)
        classes = info.features._feature_dict['label'].num_classes
        train_batches = len(train_ds)
        val_batches = len(val_ds)

        dataset_folds.append((train_ds, val_ds))
    else:
        train_ds, val_ds, info = __generate_datasets_from_tfds(dataset_name, samples_limit, batch_size, val_size, data_augmentation, augment_on_gpu)

        image_shape = info.features.shape['image']
        classes = info.features._feature_dict['label'].num_classes
        train_batches = len(train_ds)
        val_batches = len(val_ds)

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
