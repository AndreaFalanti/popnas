import os
from configparser import ConfigParser
from contextlib import redirect_stdout, redirect_stderr
from logging import Logger
from typing import Union, Tuple

import numpy as np
import pandas as pd
import psutil

from aMLLibrary import sequence_data_processing, regressor
from predictor import Predictor
from utils.feature_analysis import save_feature_analysis_plots
from utils.func_utils import strip_unused_amllibrary_config_sections
from utils.stream_to_logger import StreamToLogger


class AMLLibraryPredictor(Predictor):
    def __init__(self, config_path: str, techniques: 'list[str]', logger: Logger, log_folder: str, name: str = None, override_logs: bool = True,
                 threads: int = -1, perform_feature_analysis: int = True):
        # generate a relevant name if not set
        if name is None:
            name = f'aMLLibrary_{"_".join(techniques)}'

        super().__init__(logger, log_folder, name, override_logs)

        self.config_path = config_path
        self.techniques = techniques
        self._redir_logger = StreamToLogger(logger)

        # set default number of threads to the number of physical cores.
        # since the algorithms are very data intensive and how the GIL mechanism limits python to prefer multiprocessing
        # (pool actually spawns processes, so threads name is actually misleading), using only physical cores is preferred.
        # read more at: https://stackoverflow.com/questions/40217873/multiprocessing-use-only-the-physical-cores
        threads_div = 1 if psutil.cpu_count(logical=True) == psutil.cpu_count(logical=False) else 2
        self.threads = (len(psutil.Process().cpu_affinity()) // threads_div) if threads <= 0 else threads

        self.feature_names = None
        self.y_col = None
        self.drop_columns = None

        self.perform_feature_analysis = perform_feature_analysis

    def __setup_features_data(self, df: pd.DataFrame):
        # y to predict is always the first column in POPNAS case
        self.y_col = df.columns.values.tolist()[0]
        self.drop_columns = [self.y_col, 'exploration', 'data_augmented']
        df = df.drop(columns=self.drop_columns, errors='ignore')

        self.feature_names = df.columns.values.tolist()

    def __get_max_b(self, df: pd.DataFrame):
        return df['blocks'].max()

    def __prepare_config_file(self, dataset_path: str):
        config = ConfigParser()
        # to keep casing in keys while reading / writing
        config.optionxform = str

        config.read(self.config_path)

        strip_unused_amllibrary_config_sections(config, self.techniques)

        # value in .ini must be a single string of format ['technique1', 'technique2', ...]
        # note: '' are important for correct execution (see map)
        techniques_iter = map(lambda s: f"'{s}'", self.techniques)
        techniques_str = f"[{', '.join(techniques_iter)}]"
        config['General']['techniques'] = techniques_str
        config['General']['y'] = f'"{self.y_col}"'
        config['DataPreparation']['input_path'] = dataset_path

        save_path = os.path.join(self.log_folder, 'config.ini')
        with open(save_path, 'w') as f:
            config.write(f)

        return save_path

    def train_ensemble(self, dataset: Union[str, 'list[tuple]'], splits: int = 5, use_data_augmentation=True):
        raise NotImplementedError("aMLLibrary predictor doesn't support ensemble")

    def train(self, dataset: Union[str, 'list[Tuple]'], use_data_augmentation=True):
        if not isinstance(dataset, str):
            raise TypeError('aMLLibrary supports only files, conversion to file is a TODO...')

        dataset_df = pd.read_csv(dataset)
        actual_b = self.__get_max_b(dataset_df)
        if self.feature_names is None:
            self.__setup_features_data(dataset_df)

        train_config = self.__prepare_config_file(dataset)
        output_folder = os.path.join(self.log_folder, f'B{actual_b}')

        self._logger.info("Running regressors training on %d threads", self.threads)
        with redirect_stdout(self._redir_logger):
            with redirect_stderr(self._redir_logger):
                sequence_data_processor = sequence_data_processing.SequenceDataProcessing(train_config, output=output_folder, j=self.threads)
                best_regressor = sequence_data_processor.process()

        self.model = best_regressor  # type: regressor

        if self.perform_feature_analysis and set(self.techniques).issubset({'NNLS', 'LRRidge'}):
            features_df = dataset_df.drop(columns=self.drop_columns, errors='ignore')
            save_feature_analysis_plots(self.model.get_regressor(), features_df, output_folder, save_pred_every=500, model_type='linear')

    def predict(self, x: list) -> float:
        features_df = pd.DataFrame([x], columns=self.feature_names)
        return self.model.predict(features_df)[0]

    def predict_batch(self, x: 'list[list]') -> 'list[float]':
        features_df = pd.DataFrame(x, columns=self.feature_names)
        preds = self.model.predict(features_df)     # type: np.ndarray
        return preds.reshape(-1)

    def prepare_prediction_test_data(self, file_path: str) -> 'tuple[list[Union[str, list[tuple]]], list[list], list[list[float]]]':
        dataset_df = pd.read_csv(file_path)
        max_b = self.__get_max_b(dataset_df)
        self.__setup_features_data(dataset_df)

        dataset_paths = []
        prediction_samples = []
        real_values = []

        for b in range(1, max_b):
            train_df = dataset_df[dataset_df['blocks'] <= b]
            save_path = os.path.join(self.log_folder, f'inputs_B{b}.csv')
            train_df.to_csv(save_path, index=False)
            dataset_paths.append(save_path)

            predictions_df = dataset_df[dataset_df['blocks'] == (b + 1)]
            # in datasets that use data augmentation, filter them in prediction phase
            if 'data_augmented' in predictions_df.columns:
                # use == False, otherwise it will not work properly
                predictions_df = predictions_df[dataset_df['data_augmented'] == False]

            real_values.append(predictions_df[self.y_col].values.tolist())
            predictions_df = predictions_df.drop(columns=self.drop_columns, errors='ignore')
            prediction_samples.append(predictions_df.values.tolist())

        return dataset_paths, prediction_samples, real_values
