import os
from logging import Logger
from typing import Union, Tuple

import catboost
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform

from predictor import Predictor
from utils.feature_analysis import save_feature_analysis_plots
from utils.func_utils import create_empty_folder


class CatBoostPredictor(Predictor):
    def __init__(self, column_desc_path: str, logger: Logger, log_folder: str, name: str = None, override_logs: bool = True,
                 use_random_search: bool = False, task_type: str = 'CPU', perform_feature_analysis: bool = True):
        # generate a relevant name if not set
        if name is None:
            name = f'CatBoost_{task_type}_{"rs" if use_random_search else "default"}'

        super().__init__(logger, log_folder, name, override_logs)

        self.column_desc_path = column_desc_path

        self.use_random_search = use_random_search
        self.perform_feature_analysis = perform_feature_analysis
        self.task_type = task_type

        self.feature_names = None
        self.drop_columns = None
        self.y_col = None
        self.drop_column_indexes = None
        self.cat_indexes = None

    def __get_max_b(self, df: pd.DataFrame):
        return df['blocks'].max()

    def __setup_features_data(self, df: pd.DataFrame):
        column_desc_df = pd.read_csv(self.column_desc_path, sep='\t', names=['index', 'type'])
        self.drop_column_indexes = column_desc_df[(column_desc_df['type'] == 'Label') | (column_desc_df['type'] == 'Auxiliary')]['index']
        self.cat_indexes = column_desc_df[column_desc_df['type'] == 'Categ']['index'].tolist()
        # categorical indexes are based on whole file, since first column is the label, indexes must be decreased by 1 when using only features
        self.cat_indexes = [i - 1 for i in self.cat_indexes] if len(self.cat_indexes) else None

        # y to predict is always the first column in POPNAS case
        self.y_col = df.columns.values.tolist()[0]
        self.drop_columns = [col for i, col in enumerate(df.columns.values.tolist()) if i in self.drop_column_indexes]

        df = df.drop(columns=self.drop_columns)
        self.feature_names = df.columns.values.tolist()

    def train_ensemble(self, dataset: Union[str, 'list[tuple]'], splits: int = 5, use_data_augmentation=True):
        raise NotImplementedError("CatBoost predictor doesn't support ensemble")

    def train(self, dataset: Union[str, 'list[Tuple]'], use_data_augmentation=True):
        if not isinstance(dataset, str):
            raise TypeError('CatBoost supports only files, conversion to file is a TODO...')

        dataset_df = pd.read_csv(dataset)
        actual_b = self.__get_max_b(dataset_df)
        if self.feature_names is None:
            self.__setup_features_data(dataset_df)

        train_pool = catboost.Pool(dataset, delimiter=',', has_header=True, column_description=self.column_desc_path)
        train_log_folder = os.path.join(self.log_folder, f'B{actual_b}')
        create_empty_folder(train_log_folder)

        # Increase weight in scoring decisions of: [cells, op_score, concat_inputs], since they are the main driving factors
        # in significant time alterations.
        feature_weights = '1:1.5,2:1.5,7:1.2'
        # NOTE: task type = 'GPU' is very slow in our case, because it uses ordered sampling on datasets with few samples (< 10k).
        #  CPU is very fast and the models between GPU and CPU seems to not have so much different results
        self.model = catboost.CatBoostRegressor(iterations=2500, early_stopping_rounds=50, train_dir=train_log_folder, task_type=self.task_type,
                                                feature_weights=feature_weights)

        # train the model with random search
        if self.use_random_search:
            param_grid = {
                'learning_rate': uniform(0.02, 0.2),
                'depth': randint(3, 7),
                'l2_leaf_reg': uniform(0.1, 5),
                'random_strength': uniform(0.3, 3),
                'bagging_temperature': uniform(0.3, 3),
                # 'grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide']
            }

            results_dict = self.model.randomized_search(param_grid, train_pool, cv=5, n_iter=40, train_size=0.8)
            self._logger.info('CatBoost random search complete')
            self._logger.info('Best parameters: %s', str(results_dict['params']))
        # else simply train the model with default parameters
        else:
            self.model.fit(train_pool)
            self._logger.info('CatBoost training complete')

        if self.perform_feature_analysis:
            result_pairs = self.model.get_feature_importance(train_pool, prettified=True)  # type: pd.DataFrame
            result_pairs.to_csv(os.path.join(train_log_folder, 'feature_importance.csv'))

            features_df = dataset_df.drop(columns=self.drop_columns)
            save_feature_analysis_plots(self.model, features_df, train_log_folder, save_pred_every=60)

    def predict(self, x: list) -> float:
        # make sure categorical features are integers (and not float, pandas converts them to floats like 1.0)
        if self.cat_indexes is not None:
            for cat_index in self.cat_indexes:
                x[cat_index] = int(x[cat_index])

        features_pool = catboost.Pool([x], cat_features=self.cat_indexes, feature_names=self.feature_names)
        # returned as a numpy array of single element
        return self.model.predict(features_pool)[0]

    def predict_batch(self, x: 'list[list]') -> 'list[float]':
        # make sure categorical features are integers (and not float, pandas converts them to floats like 1.0)
        if self.cat_indexes is not None:
            for sample in x:
                for cat_index in self.cat_indexes:
                    sample[cat_index] = int(sample[cat_index])

        features_pool = catboost.Pool(x, cat_features=self.cat_indexes, feature_names=self.feature_names)
        # returned as a numpy array of single element
        preds = self.model.predict(features_pool)  # type: np.ndarray
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
            predictions_df = predictions_df.drop(columns=self.drop_columns)
            prediction_samples.append(predictions_df.values.tolist())

        return dataset_paths, prediction_samples, real_values
