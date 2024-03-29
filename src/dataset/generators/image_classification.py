import math
import os.path
from typing import Optional

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import datasets, Sequential
from tensorflow.keras.utils import image_dataset_from_directory

from dataset.augmentation import get_image_classification_tf_data_aug
from dataset.generators.base import BaseDatasetGenerator, AutoShardPolicy, SEED, DatasetsFold, generate_dataset_from_tfds, \
    generate_train_val_datasets_from_tfds
from dataset.preprocessing import ImagePreprocessor
from utils.config_dataclasses import DatasetConfig
from utils.func_utils import filter_none


class ImageClassificationDatasetGenerator(BaseDatasetGenerator):
    def __init__(self, dataset_config: DatasetConfig, optimize_for_xla_compilation: bool = False):
        super().__init__(dataset_config, optimize_for_xla_compilation)

        resize_config = dataset_config.resize
        self.resize_dim = None if resize_config is None else (resize_config.height, resize_config.width)
        self.use_cutout = dataset_config.data_augmentation.use_cutout

    def supports_early_batching(self) -> bool:
        return True

    def build_training_preprocessor(self, num_classes: int, input_size: 'tuple[int, int]', upsample_factor: float):
        '''
        Build the ImagePreprocessor for the training split of image classification tasks.
        It differs from the preprocessors of other splits because it upsamples the image by the specified factor, so that the data augmentation
        can be performed through random cropping (the pipeline acts as a zoom).

        Args:
            num_classes: the number of classes in the dataset
            input_size: image size without channel dimension (HW only)
            upsample_factor: factor > 1 to rescale the image
        '''
        # upsample image of upsample_factor in both dimensions, so that random crop to original size can be used as data augmentation
        resize_dim = tuple(int(dim * upsample_factor) for dim in input_size)
        return ImagePreprocessor(resize_dim, rescaling=(1. / 255, 0),
                                 to_one_hot=num_classes, remap_classes=self.class_labels_remapping_dict)

    def build_preprocessor(self, num_classes: int):
        '''
        Build the ImagePreprocessor for image classification tasks.

        Args:
            num_classes: the number of classes in the dataset
        '''
        return ImagePreprocessor(self.resize_dim, rescaling=(1. / 255, 0),
                                 to_one_hot=num_classes, remap_classes=self.class_labels_remapping_dict)

    def __load_keras_dataset_images(self):
        '''
        Load images of a Keras dataset. In this case, the data are in the form of Numpy arrays.

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

    def generate_train_val_datasets(self) -> 'tuple[list[DatasetsFold], int, tuple[int, ...], int, int, Optional[Sequential]]':
        dataset_folds = []  # type: list[tuple[tf.data.Dataset, tf.data.Dataset]]
        keras_aug = None
        preprocessing_model = None

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
                    train_ds = image_dataset_from_directory(os.path.join(self.dataset_path, 'keras_training'), label_mode='int',
                                                            image_size=self.resize_dim, batch_size=None, seed=SEED)
                    val_ds = image_dataset_from_directory(os.path.join(self.dataset_path, 'keras_validation'), label_mode='int',
                                                          image_size=self.resize_dim, batch_size=None, seed=SEED)

                    # limit only train dataset, if limit is set
                    if self.samples_limit is not None:
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

                        train_ds = train_ds.take(training_samples)
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
                # EuroSAT have only the train split, but conventionally the last 20% of samples is used as the test set
                # (27000 total samples, 21600 for train, last 5400 samples for test)
                if self.dataset_name == 'eurosat/rgb':
                    samples_limit = min(filter_none(21600, self.samples_limit), default=None)
                else:
                    samples_limit = self.samples_limit

                train_ds, val_ds, info = generate_train_val_datasets_from_tfds(self.dataset_name, samples_limit, self.val_size)

                image_shape = info.features['image'].shape if self.resize_dim is None else self.resize_dim + (info.features['image'].shape[2],)
                classes = info.features['label'].num_classes

            # finalize dataset generation, common logic to all dataset formats
            image_res = image_shape[0:2]
            train_preprocessor = self.build_training_preprocessor(classes, image_res, upsample_factor=1.125 if self.use_data_augmentation else 1)
            data_preprocessor = self.build_preprocessor(classes)
            tf_aug = get_image_classification_tf_data_aug(image_res, self.use_cutout) if self.use_data_augmentation else None
            train_ds, train_batches = self._finalize_dataset(train_ds, self.batch_size, train_preprocessor,
                                                             keras_data_augmentation=keras_aug, tf_data_augmentation=tf_aug,
                                                             shuffle=True, fit_preprocessing_layers=True, shard_policy=shard_policy)
            val_ds, val_batches = self._finalize_dataset(val_ds, self.val_test_batch_size, data_preprocessor, shard_policy=shard_policy)
            dataset_folds.append(DatasetsFold(train_ds, val_ds))

            preprocessing_model = train_preprocessor.preprocessor_model

        self._logger.info('Dataset folds built successfully')

        # IDE is wrong, variables are always assigned since folds > 1, so at least one cycle is always executed
        return dataset_folds, classes, image_shape, train_batches, val_batches, preprocessing_model

    def generate_test_dataset(self) -> 'tuple[tf.data.Dataset, int, tuple, int]':
        shard_policy = AutoShardPolicy.DATA

        # Custom dataset, loaded with Keras utilities
        if self.dataset_path is not None:
            if self.resize_dim is None:
                raise ValueError('Image must have a set resize dimension to use a custom dataset')

            test_ds = image_dataset_from_directory(os.path.join(self.dataset_path, 'keras_test'), label_mode='int',
                                                   image_size=self.resize_dim, batch_size=None, seed=SEED)

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
            # EuroSAT uses the last 20% of samples as test, it isn't directly separated in multiple splits
            if self.dataset_name == 'eurosat/rgb':
                split_name = 'train[21600:]'
            else:
                split_name = 'test'

            test_ds, info = generate_dataset_from_tfds(self.dataset_name, split_name, samples_limit=None)

            image_shape = info.features['image'].shape if self.resize_dim is None else self.resize_dim + (info.features['image'].shape[2],)
            classes = info.features['label'].num_classes

        # finalize dataset generation, common logic to all dataset formats
        data_preprocessor = self.build_preprocessor(classes)
        test_ds, batches = self._finalize_dataset(test_ds, self.val_test_batch_size, data_preprocessor, shard_policy=shard_policy)

        self._logger.info('Test dataset built successfully')
        return test_ds, classes, image_shape, batches

    def generate_final_training_dataset(self) -> 'tuple[tf.data.Dataset, int, tuple[int, ...], int, Optional[Sequential]]':
        keras_aug = None
        shard_policy = AutoShardPolicy.DATA

        # Custom dataset, loaded with Keras utilities
        if self.dataset_path is not None:
            if self.resize_dim is None:
                raise ValueError('Images must have a set resize dimension to use a custom dataset')

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
            # EuroSAT uses the last 20% of samples as test, it isn't directly separated in multiple splits. Use the first 80% as train+val.
            if self.dataset_name == 'eurosat/rgb':
                split_name = 'train'
                samples_limit = min(filter_none(21600, self.samples_limit), default=None)
            else:
                split_name = 'train+validation' if self.val_size is None else 'train'
                samples_limit = self.samples_limit

            train_ds, info = generate_dataset_from_tfds(self.dataset_name, split_name, samples_limit, shuffle=True)

            image_shape = info.features['image'].shape if self.resize_dim is None else self.resize_dim + (info.features['image'].shape[2],)
            classes = info.features['label'].num_classes

        # finalize dataset generation, common logic to all dataset formats
        image_res = image_shape[0:2]
        train_preprocessor = self.build_training_preprocessor(classes, image_res, upsample_factor=1.125 if self.use_data_augmentation else 1)
        tf_aug = get_image_classification_tf_data_aug(image_res, self.use_cutout) if self.use_data_augmentation else None
        train_ds, train_batches = self._finalize_dataset(train_ds, self.batch_size, train_preprocessor,
                                                         keras_data_augmentation=keras_aug, tf_data_augmentation=tf_aug,
                                                         shuffle=True, fit_preprocessing_layers=True, shard_policy=shard_policy)

        self._logger.info('Final training dataset built successfully')
        return train_ds, classes, image_shape, train_batches, train_preprocessor.preprocessor_model
