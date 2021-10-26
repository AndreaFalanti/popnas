from abc import abstractmethod
from logging import Logger
from typing import Union

import pandas as pd

from predictors import Predictor
from utils.func_utils import parse_cell_structures
from utils.rstr import rstr


class NNPredictor(Predictor):
    def __init__(self, y_col: str, y_domain: 'tuple[float, float]', logger: Logger, log_folder: str, name: str = None, epochs: int = 15,
                 use_previous_data: bool = True):
        super().__init__(logger, log_folder, name)

        self.y_col = y_col
        self.epochs = epochs
        self.use_previous_data = use_previous_data
        self.callbacks = []

        # used to accumulate samples in a common dataset (a list for each B), if use_previous_data is True
        self.children_history = []
        self.score_history = []

        # choose the correct activation for last layer, based on y domain
        self.output_activation = None
        lower_bound, upper_bound = y_domain
        assert lower_bound < upper_bound, 'Invalid domain'

        # from tests relu is bad for (0, inf) domains. If you want to do more tests, check infinite bounds with math.isinf()
        if lower_bound == 0 and upper_bound == 1:
            self.output_activation = 'sigmoid'
        elif lower_bound == -1 and upper_bound == 1:
            self.output_activation = 'tanh'
        else:
            self.output_activation = 'linear'

        self._logger.debug('Using %s as final activation, based on y domain provided', self.output_activation)

    @abstractmethod
    def _build_model(self):
        pass

    @abstractmethod
    def _build_tf_dataset(self, cell_specs: 'list[list]', rewards: 'list[float]' = None, use_data_augmentation: bool = True):
        pass

    def _get_max_b(self, df: pd.DataFrame):
        return df['# blocks'].max()

    def _extrapolate_samples_for_b(self, training_data_df: pd.DataFrame, b: int):
        b_df = training_data_df[training_data_df['# blocks'] == b]

        cells = parse_cell_structures(b_df['cell structure'])

        # just return two lists: one with the target, one with the cell structures
        return b_df[self.y_col].tolist(), cells

    def train(self, dataset: Union[str, 'list[tuple]'], use_data_augmentation=True):
        # TODO
        if not isinstance(dataset, list):
            raise TypeError('NN supports only samples, conversion from file is a TODO...')

        cells, rewards = zip(*dataset)

        # create the dataset using also previous data, if flag is set.
        # a list of values is stored for both cells and their rewards.
        if self.use_previous_data:
            self.children_history.extend(cells)
            self.score_history.extend(rewards)

            tf_dataset = self._build_tf_dataset(self.children_history, self.score_history, use_data_augmentation)
        # use only current data
        else:
            tf_dataset = self._build_tf_dataset(cells, rewards, use_data_augmentation)

        train_size = len(tf_dataset) * self.epochs
        self._logger.info("Controller: Number of training steps required for this stage : %d", train_size)

        # Controller starts from the weights learned from previous training sessions, since it's not re-instantiated.
        hist = self.model.fit(x=tf_dataset,
                              epochs=self.epochs,
                              callbacks=self.callbacks)
        self._logger.info("losses: %s", rstr(hist.history['loss']))

    def predict(self, sample: list) -> float:
        pred_dataset = self._build_tf_dataset([sample])
        return self.model.predict(x=pred_dataset)[0, 0]

    def prepare_prediction_test_data(self, file_path: str) -> 'tuple[list[Union[str, list[tuple]]], list[list], list[list[float]]]':
        dataset_df = pd.read_csv(file_path)
        max_b = self._get_max_b(dataset_df)
        drop_columns = [col for col in dataset_df.columns.values.tolist() if col not in [self.y_col, 'cell structure', '# blocks']]
        dataset_df = dataset_df.drop(columns=drop_columns)

        datasets = []
        prediction_samples = []
        real_values = []

        for b in range(1, max_b):
            targets, cells = self._extrapolate_samples_for_b(dataset_df, b)
            datasets.append(list(zip(cells, targets)))

            true_values, cells_to_predict = self._extrapolate_samples_for_b(dataset_df, b + 1)

            real_values.append(true_values)
            prediction_samples.append(cells_to_predict)

        return datasets, prediction_samples, real_values
