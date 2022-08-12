import math
import os.path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
from tensorflow.keras import datasets
from tensorflow.keras.utils import image_dataset_from_directory

from dataset.augmentation import get_image_data_augmentation_model
from dataset.generators.base import BaseDatasetGenerator, AutoShardPolicy
from dataset.preprocessing import ImagePreprocessor


class ImageClassificationDatasetGenerator(BaseDatasetGenerator):
    def __init__(self, dataset_config: dict):
        super().__init__(dataset_config)

        resize_config = dataset_config['resize']
        self.resize_dim = (resize_config['height'], resize_config['width']) if resize_config['enabled'] else None

    def __load_keras_dataset_images(self):
        '''
        Load images of a Keras dataset. In this case the data is in form of Numpy arrays.

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
        split_spec = 'train' if self.samples_limit is None else f'train[:{self.samples_limit}]'
        train_ds, info = tfds.load(self.dataset_name, split=split_spec, as_supervised=True, shuffle_files=True, with_info=True,
                                   read_config=tfds.ReadConfig(try_autocache=False))  # type: tf.data.Dataset, tfds.core.DatasetInfo

        if self.val_size is None:
            val_split_spec = 'validation' if self.samples_limit is None else f'validation[:{self.samples_limit}]'
            val_ds = tfds.load(self.dataset_name, split=val_split_spec, as_supervised=True, shuffle_files=True,
                               read_config=tfds.ReadConfig(try_autocache=False))  # type: tf.data.Dataset
        else:
            samples_count = self.samples_limit or info.splits['train'].num_examples
            train_samples = math.ceil(samples_count * (1 - self.val_size))

            val_ds = train_ds.skip(train_samples)
            train_ds = train_ds.take(train_samples)

        return train_ds, val_ds, info

    def generate_train_val_datasets(self) -> 'tuple[list[tuple[tf.data.Dataset, tf.data.Dataset]], int, tuple[int, ...], int, int]':
        dataset_folds = []  # type: list[tuple[tf.data.Dataset, tf.data.Dataset]]
        data_augmentation = get_image_data_augmentation_model() if self.use_data_augmentation else None

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

                # TODO: samples limit is applied but in a naive way, since unbatching breakes the dataset cardinality.
                #  Stratification is done due stochasticity, which is not good for low amount of samples.
                #  Anyway not a big problem since in "real" runs you would not limit the samples deliberately.
                # TODO: new tf versions make batching optional, making possible to batch the dataset in finalize like the others.
                if self.val_size is None:
                    train_ds = image_dataset_from_directory(os.path.join(self.dataset_path, 'keras_training'), label_mode='categorical',
                                                            image_size=self.resize_dim, batch_size=self.batch_size)
                    val_ds = image_dataset_from_directory(os.path.join(self.dataset_path, 'keras_validation'), label_mode='categorical',
                                                          image_size=self.resize_dim, batch_size=self.batch_size)

                    # limit only train dataset, if limit is set
                    if self.samples_limit is not None:
                        self._logger.info('Limiting training dataset to %d batches', self.samples_limit // self.batch_size)
                        train_ds = train_ds.take(self.samples_limit // self.batch_size)
                # extract a validation split from training samples
                else:
                    # TODO: find a way to make the split stratified (Stratification is done due stochasticity right now)
                    train_ds = image_dataset_from_directory(os.path.join(self.dataset_path, 'keras_training'), validation_split=self.val_size,
                                                            seed=123, subset='training', label_mode='categorical',
                                                            image_size=self.resize_dim, batch_size=self.batch_size)
                    val_ds = image_dataset_from_directory(os.path.join(self.dataset_path, 'keras_training'), validation_split=self.val_size,
                                                          seed=123, subset='validation', label_mode='categorical',
                                                          image_size=self.resize_dim, batch_size=self.batch_size)

                    # limit both datasets, so that the sum of samples is the declared one
                    if self.samples_limit is not None:
                        training_samples = math.ceil(self.samples_limit * (1 - self.val_size) // self.batch_size)
                        val_samples = math.floor(self.samples_limit * self.val_size // self.batch_size)

                        self._logger.info('Limiting training dataset to %d batches', training_samples)
                        train_ds = train_ds.take(training_samples)
                        self._logger.info('Limiting validation dataset to %d batches', val_samples)
                        val_ds = val_ds.take(val_samples)

                # TODO: sharding should be set to FILE here? or not?
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
                x_train, x_val, y_train, y_val = train_test_split(x_train_init, y_train_init, test_size=self.val_size, stratify=y_train_init)
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

            # finalize dataset generation, common logic to all dataset formats
            # avoid categorical for custom datasets, since already done
            using_custom_ds = self.dataset_path is not None
            data_preprocessor = ImagePreprocessor(self.resize_dim, rescaling=(1. / 255, 0), to_one_hot=None if using_custom_ds else classes)
            # TODO: remove when updating TF to new version, since custom dataset is not forced to be batched
            batch_len = None if using_custom_ds else self.batch_size
            train_ds, train_batches = self._finalize_dataset(train_ds, batch_len, data_preprocessor, data_augmentation, shard_policy=shard_policy)
            val_ds, val_batches = self._finalize_dataset(val_ds, batch_len, data_preprocessor, None, shard_policy=shard_policy)
            dataset_folds.append((train_ds, val_ds))

        self._logger.info('Dataset folds built successfully')

        # IDE is wrong, variables are always assigned since folds > 1, so at least one cycle is always executed
        return dataset_folds, classes, image_shape, train_batches, val_batches

    def generate_test_dataset(self) -> 'tuple[tf.data.Dataset, int, tuple, int]':
        shard_policy = AutoShardPolicy.DATA

        # Custom dataset, loaded with Keras utilities
        if self.dataset_path is not None:
            if self.resize_dim is None:
                raise ValueError('Image must have a set resize dimension to use a custom dataset')

            # TODO: new tf versions make batching optional, making possible to batch the dataset in finalize like the others.
            test_ds = image_dataset_from_directory(os.path.join(self.dataset_path, 'keras_test'), label_mode='categorical',
                                                   image_size=self.resize_dim, batch_size=self.batch_size)

            # TODO: sharding should be set to FILE here? or not?
            shard_policy = AutoShardPolicy.FILE

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
            test_ds, info = tfds.load(self.dataset_name, split='test', as_supervised=True, shuffle_files=False, with_info=True,
                                      read_config=tfds.ReadConfig(try_autocache=False))  # type: tf.data.Dataset, tfds.core.DatasetInfo

            image_shape = info.features['image'].shape if self.resize_dim is None else self.resize_dim + (info.features['image'].shape[2],)
            classes = info.features['label'].num_classes

        # finalize dataset generation, common logic to all dataset formats
        # avoid categorical for custom datasets, since already done
        using_custom_ds = self.dataset_path is not None
        data_preprocessor = ImagePreprocessor(self.resize_dim, rescaling=(1. / 255, 0), to_one_hot=None if using_custom_ds else classes)
        # TODO: remove when updating TF to new version, since custom dataset is not forced to be batched
        batch_len = None if using_custom_ds else self.batch_size
        test_ds, batches = self._finalize_dataset(test_ds, batch_len, data_preprocessor, None, shard_policy=shard_policy)

        self._logger.info('Test dataset built successfully')
        return test_ds, classes, image_shape, batches
