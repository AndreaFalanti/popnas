import math
import os.path
from typing import Optional

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from albumentations import Compose
from sklearn.model_selection import train_test_split
from tensorflow.keras import datasets, Sequential
from tensorflow.keras.utils import image_dataset_from_directory

from dataset.generators.base import BaseDatasetGenerator, AutoShardPolicy, SEED
from dataset.transformations.augmentation import get_image_classification_augmentations
from dataset.transformations.preprocessing import labels_to_one_hot, preprocess_image


def generate_preprocessing_function(classes: int, resize_dim: tuple[int, int]):
    ''' Generates a closure, which can be mapped to the TF dataset for preprocessing. '''
    transforms = preprocess_image(resize_dim, image_to_float=True)

    def preprocess_x(x):
        transformed = transforms(image=x)
        return transformed['image']

    def preprocessing_f(x, y):
        y = labels_to_one_hot(classes)(y)
        x = tf.numpy_function(preprocess_x, inp=[x], Tout=tf.float32)
        return x, y

    return preprocessing_f


def generate_augmentation_function(transforms: Compose):
    def augment_x(x):
        transformed = transforms(image=x)
        return transformed['image']

    def augmentation_f(x, y):
        x = tf.numpy_function(augment_x, inp=[x], Tout=tf.float32)
        return x, y

    return augmentation_f


class ImageClassificationDatasetGenerator(BaseDatasetGenerator):
    def __init__(self, dataset_config: dict, enable_tpu_tricks: bool = False):
        super().__init__(dataset_config, enable_tpu_tricks)

        resize_config = dataset_config['resize']
        self.resize_dim = (resize_config['height'], resize_config['width']) if resize_config['enabled'] else None
        self.use_cutout = dataset_config['data_augmentation'].get('use_cutout', False)

        if self.use_data_augmentation:
            transforms = get_image_classification_augmentations(self.use_cutout)
            self.augmentation_f = generate_augmentation_function(transforms)
        else:
            self.augmentation_f = None

    def __load_keras_dataset_images(self):
        '''
        Load images of a Keras dataset, which consists of Numpy arrays.

        Returns:
            train images, test images and the classes contained in the dataset
        '''
        if self.dataset_name == 'cifar10':
            (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
            classes_count = 10
            channels = 3
        elif self.dataset_name == 'cifar100':
            (x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()
            classes_count = 100
            channels = 3
        elif self.dataset_name == 'fashion_mnist':
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

    def __generate_datasets_from_tfds(self):
        # EuroSAT have only train split, but conventionally the last 20% of samples is used as test set
        # (27000 total, 21600 for train, last 5400 for test)
        if self.dataset_name == 'eurosat/rgb':
            split_spec = 'train[:21600]' if self.samples_limit is None else f'train[:{min(21600, self.samples_limit)}]'
        else:
            split_spec = 'train' if self.samples_limit is None else f'train[:{self.samples_limit}]'

        train_ds, info = tfds.load(self.dataset_name, split=split_spec, as_supervised=True, shuffle_files=True, with_info=True,
                                   read_config=tfds.ReadConfig(try_autocache=False))  # type: tf.data.Dataset, tfds.core.DatasetInfo

        if self.val_size is None:
            val_split_spec = 'validation' if self.samples_limit is None else f'validation[:{self.samples_limit}]'
            val_ds = tfds.load(self.dataset_name, split=val_split_spec, as_supervised=True, shuffle_files=True,
                               read_config=tfds.ReadConfig(try_autocache=False))  # type: tf.data.Dataset
        else:
            samples_count = self.samples_limit or info.splits['train'].num_examples
            if self.dataset_name == 'eurosat/rgb':
                samples_count = 21600

            train_samples = math.ceil(samples_count * (1 - self.val_size))

            val_ds = train_ds.skip(train_samples)
            train_ds = train_ds.take(train_samples)

        return train_ds, val_ds, info

    def generate_train_val_datasets(
            self) -> 'tuple[list[tuple[tf.data.Dataset, tf.data.Dataset]], int, tuple[int, ...], int, int, Optional[Sequential]]':
        dataset_folds = []  # type: list[tuple[tf.data.Dataset, tf.data.Dataset]]

        # TODO: folds were implemented only in Keras datasets, later i have externalized the logic to support the other dataset format. Still,
        #  the train-validation is done in different ways based on format right now, and if seed is fixed the folds could be identical.
        #  Folds right now are never used, if this feature become relevant check that folds are actually stochastic, or use K-fold et simila.
        for i in range(self.dataset_folds_count):
            self._logger.info('Preprocessing and building dataset fold #%d...', i + 1)
            shard_policy = AutoShardPolicy.DATA

            # Custom dataset, loaded with Keras utilities
            if self.dataset_path is not None:
                if self.resize_dim is None:
                    raise ValueError('Image must have a set resize dimension to use a custom dataset')

                if self.val_size is None:
                    train_ds = image_dataset_from_directory(os.path.join(self.dataset_path, 'keras_training'), label_mode='categorical',
                                                            image_size=self.resize_dim, batch_size=self.batch_size, seed=SEED)
                    val_ds = image_dataset_from_directory(os.path.join(self.dataset_path, 'keras_validation'), label_mode='categorical',
                                                          image_size=self.resize_dim, batch_size=self.batch_size, seed=SEED)

                    # limit only train dataset, if limit is set
                    if self.samples_limit is not None:
                        self._logger.info('Limiting training dataset to %d batches', self.samples_limit)
                        train_ds = train_ds.take(self.samples_limit)
                # extract a validation split from training samples
                else:
                    # TODO: find a way to make the split stratified (Stratification is done due stochasticity right now)
                    train_ds = image_dataset_from_directory(os.path.join(self.dataset_path, 'keras_training'), validation_split=self.val_size,
                                                            subset='training', label_mode='int',
                                                            image_size=self.resize_dim, batch_size=None, seed=SEED)
                    val_ds = image_dataset_from_directory(os.path.join(self.dataset_path, 'keras_training'), validation_split=self.val_size,
                                                          subset='validation', label_mode='int',
                                                          image_size=self.resize_dim, batch_size=None, seed=SEED)

                    # limit both datasets, so that the sum of samples is the declared one
                    if self.samples_limit is not None:
                        training_samples = math.ceil(self.samples_limit * (1 - self.val_size))
                        val_samples = math.floor(self.samples_limit * self.val_size)

                        self._logger.info('Limiting training dataset to %d samples', training_samples)
                        train_ds = train_ds.take(training_samples)
                        self._logger.info('Limiting validation dataset to %d samples', val_samples)
                        val_ds = val_ds.take(val_samples)

                shard_policy = AutoShardPolicy.FILE
                classes = self.dataset_classes_count
                image_shape = self.resize_dim + (3,)
            # Keras dataset case
            elif self.dataset_name in ['cifar10', 'cifar100', 'fashion_mnist']:
                train, test, classes, image_shape, channels = self.__load_keras_dataset_images()
                classes = classes or self.dataset_classes_count  # like || in javascript

                x_train_init, y_train_init = train
                if self.samples_limit is not None:
                    x_train_init = x_train_init[:self.samples_limit]
                    y_train_init = y_train_init[:self.samples_limit]

                if self.resize_dim is not None:
                    image_shape = self.resize_dim + (channels,)

                # create a validation set for evaluation of the child models
                x_train, x_val, y_train, y_val = train_test_split(x_train_init, y_train_init, test_size=self.val_size, stratify=y_train_init,
                                                                  random_state=SEED)
                # remove last axis
                y_train = np.squeeze(y_train)
                y_val = np.squeeze(y_val)

                train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
                val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
            # TFDS case
            else:
                train_ds, val_ds, info = self.__generate_datasets_from_tfds()

                image_shape = info.features['image'].shape if self.resize_dim is None else self.resize_dim + (info.features['image'].shape[2],)
                classes = info.features['label'].num_classes

            # generate the data transformations
            preprocessing_f = generate_preprocessing_function(classes, self.resize_dim)

            # finalize dataset generation, common logic to all dataset formats
            train_ds, train_batches, preprocessing_model = self._finalize_dataset(train_ds, self.batch_size, shuffle=True, shard_policy=shard_policy,
                                                                                  data_preprocessing=preprocessing_f,
                                                                                  data_augmentation=self.augmentation_f)
            val_ds, val_batches, _ = self._finalize_dataset(val_ds, self.batch_size, shard_policy=shard_policy,
                                                            data_preprocessing=preprocessing_f)
            dataset_folds.append((train_ds, val_ds))

        self._logger.info('Dataset folds built successfully')

        # IDE is wrong, variables are always assigned since folds > 1, so at least one cycle is always executed
        return dataset_folds, classes, image_shape, train_batches, val_batches, preprocessing_model

    def generate_test_dataset(self) -> 'tuple[tf.data.Dataset, int, tuple, int]':
        shard_policy = AutoShardPolicy.DATA

        # Custom dataset, loaded with Keras utilities
        if self.dataset_path is not None:
            if self.resize_dim is None:
                raise ValueError('Image must have a set resize dimension to use a custom dataset')

            shard_policy = AutoShardPolicy.FILE
            test_ds = image_dataset_from_directory(os.path.join(self.dataset_path, 'keras_test'), label_mode='int',
                                                   image_size=self.resize_dim, batch_size=None, seed=SEED)

            classes = self.dataset_classes_count
            image_shape = self.resize_dim + (3,)
        # Keras dataset case
        elif self.dataset_name in ['cifar10', 'cifar100', 'fashion_mnist']:
            train, test, classes, image_shape, channels = self.__load_keras_dataset_images()
            classes = classes or self.dataset_classes_count  # like || in javascript

            x_test, y_test = test

            if self.resize_dim is not None:
                image_shape = self.resize_dim + (channels,)

            # remove last axis
            y_test = np.squeeze(y_test)

            test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        # TFDS case
        else:
            # EuroSAT uses the last 20% of samples as test, it isn't directly separated in multiple splits
            if self.dataset_name == 'eurosat/rgb':
                split_name = 'train[21600:]'
            else:
                split_name = 'test'

            test_ds, info = tfds.load(self.dataset_name, split=split_name, as_supervised=True, shuffle_files=False, with_info=True,
                                      read_config=tfds.ReadConfig(try_autocache=False))  # type: tf.data.Dataset, tfds.core.DatasetInfo

            image_shape = info.features['image'].shape if self.resize_dim is None else self.resize_dim + (info.features['image'].shape[2],)
            classes = info.features['label'].num_classes

        # generate the data transformations
        preprocessing_f = generate_preprocessing_function(classes, self.resize_dim)
        test_ds, batches, _ = self._finalize_dataset(test_ds, self.batch_size, shard_policy=shard_policy,
                                                     data_preprocessing=preprocessing_f)

        self._logger.info('Test dataset built successfully')
        return test_ds, classes, image_shape, batches

    def generate_final_training_dataset(self) -> 'tuple[tf.data.Dataset, int, tuple[int, ...], int, Optional[Sequential]]':
        shard_policy = AutoShardPolicy.DATA

        # Custom dataset, loaded with Keras utilities
        if self.dataset_path is not None:
            if self.resize_dim is None:
                raise ValueError('Image must have a set resize dimension to use a custom dataset')

            train_ds = image_dataset_from_directory(os.path.join(self.dataset_path, 'keras_training'), label_mode='int',
                                                    image_size=self.resize_dim, batch_size=None, seed=SEED)

            # if val_size is None, it means that a validation split is present, merge the two datasets
            if self.val_size is None:
                val_ds = image_dataset_from_directory(os.path.join(self.dataset_path, 'keras_validation'), label_mode='int',
                                                      image_size=self.resize_dim, batch_size=None, seed=SEED)
                train_ds = train_ds.concatenate(val_ds)

            shard_policy = AutoShardPolicy.FILE

            classes = self.dataset_classes_count
            image_shape = self.resize_dim + (3,)
        # Keras dataset case
        elif self.dataset_name in ['cifar10', 'cifar100', 'fashion_mnist']:
            train, test, classes, image_shape, channels = self.__load_keras_dataset_images()
            classes = classes or self.dataset_classes_count  # like || in javascript

            x_train, y_train = train

            if self.resize_dim is not None:
                image_shape = self.resize_dim + (channels,)

            # remove last axis
            y_train = np.squeeze(y_train)

            train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        # TFDS case
        else:
            # EuroSAT use the last 20% of samples as test, it isn't directly separated in multiple splits. Use the first 80% as train+val.
            if self.dataset_name == 'eurosat/rgb':
                tfds_split = 'train[:21600]'
            else:
                tfds_split = 'train+validation' if self.val_size is None else 'train'

            train_ds, info = tfds.load(self.dataset_name, split=tfds_split, as_supervised=True, shuffle_files=True, with_info=True,
                                       read_config=tfds.ReadConfig(try_autocache=False))  # type: tf.data.Dataset, tfds.core.DatasetInfo

            image_shape = info.features['image'].shape if self.resize_dim is None else self.resize_dim + (info.features['image'].shape[2],)
            classes = info.features['label'].num_classes

        # generate the data transformations
        preprocessing_f = generate_preprocessing_function(classes, self.resize_dim)

        train_ds, train_batches, preprocessing_model = self._finalize_dataset(train_ds, self.batch_size, shuffle=True, shard_policy=shard_policy,
                                                                              data_preprocessing=preprocessing_f,
                                                                              data_augmentation=self.augmentation_f)

        self._logger.info('Final training dataset built successfully')
        return train_ds, classes, image_shape, train_batches, preprocessing_model
