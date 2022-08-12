import os.path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sktime import datasets as sktdata

from dataset.generators.base import BaseDatasetGenerator, AutoShardPolicy
from dataset.preprocessing import TimeSeriesPreprocessor


class TimeSeriesClassificationDatasetGenerator(BaseDatasetGenerator):
    def __init__(self, dataset_config: dict):
        super().__init__(dataset_config)

    def generate_train_val_datasets(self) -> 'tuple[list[tuple[tf.data.Dataset, tf.data.Dataset]], int, tuple[int, ...], int, int]':
        dataset_folds = []  # type: list[tuple[tf.data.Dataset, tf.data.Dataset]]

        for i in range(self.dataset_folds_count):
            self._logger.info('Preprocessing and building dataset fold #%d...', i + 1)
            shard_policy = AutoShardPolicy.DATA

            # Custom dataset, loaded from numpy arrays
            if self.dataset_path is not None:
                # numpy case
                if os.path.exists(os.path.join(self.dataset_path, 'numpy_training', 'train.npz')):
                    train_npz = np.load(os.path.join(self.dataset_path, 'numpy_training', 'train.npz'))
                    x_train, y_train = train_npz['x'], train_npz['y']
                # ts (sktime) case
                elif os.path.exists(os.path.join(self.dataset_path, 'train.ts')):
                    x_train, y_train = sktdata.load_from_tsfile(os.path.join(self.dataset_path, 'train.ts'), return_data_type='numpy3d')
                    # don't know why, but seems they prefer (num_series, ts_length) as format instead of CONV1D required (ts_length, num_series)
                    # swap the axis
                    x_train = np.swapaxes(x_train, -2, -1)
                    # cast to int, in case str is used (need intermediate conversion to float)
                    y_train = y_train.astype(np.float).astype(np.int32)
                else:
                    raise ValueError('No supported dataset format recognized at path provided in configuration')

                # make a plain list to perform one_hot correctly in preprocessor
                y_train = np.squeeze(y_train)

                classes = len(np.unique(y_train))
                # discard first dimension since is the number of samples
                input_shape = np.shape(x_train)[1:]

                if self.samples_limit is not None:
                    # also shuffle, since .ts files are usually ordered by class. Without shuffle, entire classes could be dropped.
                    x_train, y_train = shuffle(x_train, y_train, n_samples=self.samples_limit)

                # create a validation set for evaluation of the child models
                x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=self.val_size, stratify=y_train)

                train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
                val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
            else:
                raise NotImplementedError()

            preprocessor = TimeSeriesPreprocessor(to_one_hot=classes)
            train_ds, train_batches = self._finalize_dataset(train_ds, self.batch_size, preprocessor, None, shard_policy=shard_policy)
            val_ds, val_batches = self._finalize_dataset(val_ds, self.batch_size, preprocessor, None, shard_policy=shard_policy)
            dataset_folds.append((train_ds, val_ds))

        self._logger.info('Dataset folds built successfully')

        # IDE is wrong, variables are always assigned since folds > 1, so at least one cycle is always executed
        return dataset_folds, classes, input_shape, train_batches, val_batches

    def generate_test_dataset(self) -> 'tuple[tf.data.Dataset, int, tuple, int]':
        shard_policy = AutoShardPolicy.DATA

        raise NotImplementedError('Test set is not supported yet')

        # self._logger.info('Test dataset built successfully')
        # return test_ds, classes, image_shape, batches
