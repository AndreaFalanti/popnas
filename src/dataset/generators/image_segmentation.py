import os.path
from typing import Optional

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras import Sequential

from dataset.generators.base import BaseDatasetGenerator, AutoShardPolicy, SEED, load_npz, generate_tf_dataset_from_numpy_ragged_array
from dataset.preprocessing import ImagePreprocessor


class ImageSegmentationDatasetGenerator(BaseDatasetGenerator):
    def __init__(self, dataset_config: dict, enable_tpu_tricks: bool = False):
        super().__init__(dataset_config, enable_tpu_tricks)

        resize_config = dataset_config['resize']
        self.resize_dim = (resize_config['height'], resize_config['width']) if resize_config['enabled'] else None

    def generate_train_val_datasets(self) -> 'tuple[list[tuple[tf.data.Dataset, tf.data.Dataset]], int, tuple[int, ...], int, int, Optional[Sequential]]':
        dataset_folds = []  # type: list[tuple[tf.data.Dataset, tf.data.Dataset]]

        # TODO: currently not supported
        preprocessing_model = None
        keras_aug = None
        tf_aug = None
        # tf_aug = get_image_segmentation_tf_data_augmentation_functions(self.resize_dim)

        for i in range(self.dataset_folds_count):
            self._logger.info('Preprocessing and building dataset fold #%d...', i + 1)
            shard_policy = AutoShardPolicy.DATA

            # TODO: currently resizing is not working, neither random crop. There is a problem due to ragged tensors special format...
            # Custom dataset, loaded from numpy npz files
            if self.dataset_path is not None:
                classes = self.dataset_classes_count
                image_shape = self.resize_dim + (3,) if self.resize_dim else (None, None, 3)

                x_train, y_train = load_npz(os.path.join(self.dataset_path, 'train.npz'), possibly_ragged=True)
                # shuffle samples, then stratify in train-val split if needed.
                # sample limit will also be applied in case it is not None
                x_train, y_train = shuffle(x_train, y_train, n_samples=self.samples_limit, random_state=SEED)

                if self.val_size is None:
                    x_val, y_val = load_npz(os.path.join(self.dataset_path, 'validation.npz'), possibly_ragged=True)
                # extract a validation split from training samples
                else:
                    # TODO: stratification removed since not working properly on 2D labels. Find a way to add it.
                    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=self.val_size, random_state=SEED)

                # if type is object, then the dataset was saved in a ragged array format
                if x_train.dtype == np.object:
                    train_ds = generate_tf_dataset_from_numpy_ragged_array(x_train, y_train, dtype=None)
                    val_ds = generate_tf_dataset_from_numpy_ragged_array(x_train, y_train, dtype=np.uint8)
                # normal case, samples of the same dimensions are expected
                else:
                    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
                    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
            # Keras dataset case
            else:
                raise NotImplementedError('Only custom datasets are supported right now')

            # finalize dataset generation, common logic to all dataset formats
            data_preprocessor = ImagePreprocessor(self.resize_dim, rescaling=(1. / 255, 0), to_one_hot=None, resize_labels=True)
            train_ds, train_batches = self._finalize_dataset(train_ds, self.batch_size, data_preprocessor,
                                                             keras_data_augmentation=keras_aug, tf_data_augmentation_fns=tf_aug,
                                                             shuffle=True, fit_preprocessing_layers=True, shard_policy=shard_policy)
            val_ds, val_batches = self._finalize_dataset(val_ds, self.batch_size, data_preprocessor, shard_policy=shard_policy)
            dataset_folds.append((train_ds, val_ds))

            preprocessing_model = data_preprocessor.preprocessor_model

        self._logger.info('Dataset folds built successfully')

        # IDE is wrong, variables are always assigned since folds > 1, so at least one cycle is always executed
        return dataset_folds, classes, image_shape, train_batches, val_batches, preprocessing_model

    def generate_test_dataset(self) -> 'tuple[tf.data.Dataset, int, tuple, int]':
        shard_policy = AutoShardPolicy.DATA

        # Custom dataset, loaded from numpy npz files
        if self.dataset_path is not None:
            x_test, y_test = load_npz(os.path.join(self.dataset_path, 'test.npz'))

            test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

            classes = self.dataset_classes_count
            image_shape = self.resize_dim + (3,) if self.resize_dim else (None, None, 3)
        # Keras dataset case
        else:
            raise NotImplementedError('Only custom datasets are supported right now')

        # finalize dataset generation, common logic to all dataset formats
        data_preprocessor = ImagePreprocessor(self.resize_dim, rescaling=(1. / 255, 0), to_one_hot=None, resize_labels=True)
        test_ds, batches = self._finalize_dataset(test_ds, self.batch_size, data_preprocessor, shard_policy=shard_policy)

        self._logger.info('Test dataset built successfully')
        return test_ds, classes, image_shape, batches

    def generate_final_training_dataset(self) -> 'tuple[tf.data.Dataset, int, tuple[int, ...], int, Optional[Sequential]]':
        # TODO: currently not supported
        keras_aug = None
        tf_aug = None

        shard_policy = AutoShardPolicy.DATA

        # Custom dataset, loaded from numpy npz files
        if self.dataset_path is not None:
            x_train, y_train = load_npz(os.path.join(self.dataset_path, 'train.npz'))

            # merge validation split into train
            if self.val_size is None:
                x_val, y_val = load_npz(os.path.join(self.dataset_path, 'validation.npz'))
                x_train = np.concatenate((x_train, x_val), axis=0)
                y_train = np.concatenate((y_train, y_val), axis=0)

            x_train, y_train = shuffle(x_train, y_train, n_samples=self.samples_limit, random_state=SEED)
            train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))

            classes = self.dataset_classes_count
            image_shape = self.resize_dim + (3,) if self.resize_dim else (None, None, 3)
        # Keras dataset case
        else:
            raise NotImplementedError('Only custom datasets are supported right now')

        # finalize dataset generation, common logic to all dataset formats
        data_preprocessor = ImagePreprocessor(self.resize_dim, rescaling=(1. / 255, 0), to_one_hot=None, resize_labels=True)
        train_ds, train_batches = self._finalize_dataset(train_ds, self.batch_size, data_preprocessor,
                                                         keras_data_augmentation=keras_aug, tf_data_augmentation_fns=tf_aug,
                                                         shuffle=True, fit_preprocessing_layers=True, shard_policy=shard_policy)

        self._logger.info('Final training dataset built successfully')
        return train_ds, classes, image_shape, train_batches, data_preprocessor.preprocessor_model
