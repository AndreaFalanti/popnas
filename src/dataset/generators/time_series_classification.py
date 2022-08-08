import os.path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from dataset.augmentation import get_image_data_augmentation_model
from dataset.generators.base import BaseDatasetGenerator
from dataset.preprocessing import TimeSeriesPreprocessor

AUTOTUNE = tf.data.AUTOTUNE


class TimeSeriesClassificationDatasetGenerator(BaseDatasetGenerator):
    def __init__(self, dataset_config: dict):
        super().__init__(dataset_config)


    def generate_train_val_datasets(self) -> 'tuple[list[tf.data.Dataset, tf.data.Dataset], int, tuple, int, int]':
        dataset_folds = []  # type: list[tuple[tf.data.Dataset, tf.data.Dataset]]
        data_augmentation = get_image_data_augmentation_model() if self.use_data_augmentation else None

        for i in range(self.dataset_folds_count):
            self._logger.info('Preprocessing and building dataset fold #%d...', i + 1)
            shard_policy = tf.data.experimental.AutoShardPolicy.DATA

            # Custom dataset, loaded from numpy arrays
            if self.dataset_path is not None:
                x_train = np.load(os.path.join(self.dataset_path, 'numpy_training', 'X.npy'))
                y_train = np.load(os.path.join(self.dataset_path, 'numpy_training', 'Y.npy'))
                # make a plain list to perform one_hot correctly in preprocessor
                y_train = np.squeeze(y_train)

                classes = len(np.unique(y_train))
                # discard first dimension since is the number of samples
                input_shape = np.shape(x_train)[1:]

                if self.samples_limit is not None:
                    x_train = x_train[:self.samples_limit]
                    y_train = y_train[:self.samples_limit]

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
        shard_policy = tf.data.experimental.AutoShardPolicy.DATA

        raise NotImplementedError('Test set is not supported yet')

        # self._logger.info('Test dataset built successfully')
        # return test_ds, classes, image_shape, batches
