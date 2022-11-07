import os.path
from typing import Optional

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sktime import datasets as sktdata
from tensorflow.keras import Sequential

from dataset.generators.base import BaseDatasetGenerator, SEED
from dataset.preprocessing import TimeSeriesPreprocessor


def _load_npz(file_path: str) -> 'tuple[np.ndarray, np.ndarray]':
    npz = np.load(file_path)
    return npz['x'], npz['y']


def _load_ts(file_path: str) -> 'tuple[np.ndarray, np.ndarray]':
    ''' Load .ts file into numpy array, modifying the format for TF operators. '''
    x_train, y_train = sktdata.load_from_tsfile(file_path, return_data_type='numpy3d')
    # ts use array format (num_series, ts_length), instead of TF 1D ops required format (ts_length, num_series). Swap the axis.
    x_train = np.swapaxes(x_train, -2, -1)
    # ts encode labels as str (e.g. 1.0, 2.0). Cast to int for one-hot encoding (need intermediate conversion to float).
    y_train = y_train.astype(np.float).astype(np.int32)

    return x_train, y_train


def _convert_labels_to_zero_indexed_categorical(y: np.ndarray):
    class_labels = np.unique(y)     # unique already sort in ascending order
    num_classes = len(class_labels)

    # already 0-indexed labels
    if set(class_labels) == set(range(num_classes)):
        return y
    else:
        # Copy array and use original for mask purposes.
        # Doing this is possible to avoid cases where labels can be merged (example: [-1, 0], -1 are substituted with 0,
        # then 0 are substituted with 1, so all labels become 1). This approach solves the issue.
        new_y = y.copy()
        for i, label in enumerate(class_labels):
            new_y[y == label] = i

        return new_y


class TimeSeriesClassificationDatasetGenerator(BaseDatasetGenerator):
    def __init__(self, dataset_config: dict):
        super().__init__(dataset_config)

        self.rescale = dataset_config['rescale']
        self.normalize = dataset_config['normalize']

    def _get_numpy_split(self, split_name: str):
        # numpy case
        if os.path.exists(os.path.join(self.dataset_path, f'{split_name}.npz')):
            x, y = _load_npz(os.path.join(self.dataset_path, f'{split_name}.npz'))
        # ts (sktime) case
        else:
            try:
                # check if there is a TS file ending with split name (both custom and UCR-UEA archives follow this convention)
                ts_filename = [filename for filename in os.listdir(self.dataset_path) if filename.lower().endswith(f'{split_name}.ts')][0]
                x, y = _load_ts(os.path.join(self.dataset_path, ts_filename))
            except:
                raise ValueError('No supported dataset format recognized at path provided in configuration')

        # make a plain list to perform one_hot correctly in preprocessor
        y = np.squeeze(y)
        # also make sure that labels are consecutive 0-indexed integers (0 to num_classes - 1)
        y = _convert_labels_to_zero_indexed_categorical(y)

        return x, y

    def generate_train_val_datasets(self) -> 'tuple[list[tuple[tf.data.Dataset, tf.data.Dataset]], int, tuple[int, ...], int, int, Optional[Sequential]]':
        dataset_folds = []  # type: list[tuple[tf.data.Dataset, tf.data.Dataset]]

        preprocessing_model = None
        for i in range(self.dataset_folds_count):
            self._logger.info('Preprocessing and building dataset fold #%d...', i + 1)

            # Custom datasets
            if self.dataset_path is not None:
                x_train, y_train = self._get_numpy_split('train')
                q_x = np.quantile(x_train, q=0.98) if self.rescale else 1

                classes = len(np.unique(y_train))
                # discard first dimension since is the number of samples
                input_shape = np.shape(x_train)[1:]

                if self.samples_limit is not None:
                    # also shuffle, since .ts files are usually ordered by class. Without shuffle, entire classes could be dropped.
                    x_train, y_train = shuffle(x_train, y_train, n_samples=self.samples_limit, random_state=SEED)

                # create a validation set for evaluation of the child models. If val_size is None, then files for validation split must exist.
                if self.val_size is None:
                    x_val, y_val = self._get_numpy_split('validation')
                else:
                    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=self.val_size, stratify=y_train,
                                                                      random_state=SEED)

                train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
                val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
            else:
                raise NotImplementedError()

            preprocessor = TimeSeriesPreprocessor(to_one_hot=classes, normalize=self.normalize, rescale=q_x)
            train_ds, train_batches = self._finalize_dataset(train_ds, self.batch_size, preprocessor, shuffle=True, fit_preprocessing_layers=True)
            val_ds, val_batches = self._finalize_dataset(val_ds, self.batch_size, preprocessor)
            dataset_folds.append((train_ds, val_ds))

            preprocessing_model = preprocessor.preprocessor_model

        self._logger.info('Dataset folds built successfully')

        # IDE is wrong, variables are always assigned since folds > 1, so at least one cycle is always executed
        return dataset_folds, classes, input_shape, train_batches, val_batches, preprocessing_model

    def generate_test_dataset(self) -> 'tuple[tf.data.Dataset, int, tuple, int]':
        # Custom datasets
        if self.dataset_path is not None:
            x_test, y_test = self._get_numpy_split('test')
            q_x = np.quantile(x_test, q=0.98) if self.rescale else 1

            classes = len(np.unique(y_test))
            # discard first dimension since is the number of samples
            input_shape = np.shape(x_test)[1:]

            test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        else:
            raise NotImplementedError()

        preprocessor = TimeSeriesPreprocessor(to_one_hot=classes, normalize=self.normalize, rescale=q_x)
        test_ds, batches = self._finalize_dataset(test_ds, self.batch_size, preprocessor)

        self._logger.info('Test dataset built successfully')
        return test_ds, classes, input_shape, batches

    def generate_final_training_dataset(self) -> 'tuple[tf.data.Dataset, int, tuple[int, ...], int, Optional[Sequential]]':
        # Custom datasets
        if self.dataset_path is not None:
            x_train, y_train = self._get_numpy_split('train')

            classes = len(np.unique(y_train))
            # discard first dimension since is the number of samples
            input_shape = np.shape(x_train)[1:]

            # create a validation set for evaluation of the child models. If val_size is None, then files for validation split must exist.
            if self.val_size is None:
                x_val, y_val = self._get_numpy_split('validation')
                x_train = np.concatenate((x_train, x_val), axis=0)
                y_train = np.concatenate((y_train, y_val), axis=0)

            q_x = np.quantile(x_train, q=0.98) if self.rescale else 1
            train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        else:
            raise NotImplementedError()

        preprocessor = TimeSeriesPreprocessor(to_one_hot=classes, normalize=self.normalize, rescale=q_x)
        train_ds, train_batches = self._finalize_dataset(train_ds, self.batch_size, preprocessor, shuffle=True, fit_preprocessing_layers=True)

        self._logger.info('Dataset folds built successfully')
        return train_ds, classes, input_shape, train_batches, preprocessor.preprocessor_model

