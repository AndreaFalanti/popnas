import os
from logging import Logger
from typing import Union, Tuple, Sequence, Optional

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from lightgbm import early_stopping, log_evaluation
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from utils.func_utils import create_empty_folder
from .predictor import Predictor


class LGBMPredictor(Predictor):
    def __init__(self, logger: Logger, log_folder: str, name: str = None, cat_feature_names: Optional[list[str]] = None,
                 drop_feature_names : Optional[list[str]] = None, override_logs: bool = True, use_random_search: bool = False,
                 task_type: str = 'CPU', perform_feature_analysis: bool = True):
        # avoids mutable arguments, set to an empty list by default if not provided
        if drop_feature_names is None:
            drop_feature_names = []
        if cat_feature_names is None:
            cat_feature_names = []

        # generate a relevant name if not set
        if name is None:
            name = f'LGBM_{task_type}_{"rs" if use_random_search else "default"}'

        super().__init__(logger, log_folder, name, override_logs)

        self.use_random_search = use_random_search
        self.perform_feature_analysis = perform_feature_analysis
        self.task_type = task_type

        self.feature_names = None  # type: list
        self.drop_feature_names = drop_feature_names
        self.label_name = None
        self.drop_column_indexes = None
        self.cat_feature_names = cat_feature_names
        self.cat_feature_indexes = []

    def __get_max_b(self, df: pd.DataFrame):
        return df['blocks'].max()

    def __setup_features_data(self, df: pd.DataFrame):
        # y to predict is always the first column in POPNAS case
        self.label_name = df.columns.values.tolist()[0]

        # drop label and specified columns
        df = df.drop(columns=self.drop_feature_names)
        self.feature_names = df.columns.values.tolist()

        self.cat_feature_indexes = [i for i, f_name in enumerate(self.feature_names) if f_name in self.cat_feature_names]

    def __objective(self, trial: optuna.Trial, X, y):
        param_grid = {
            #         "device_type": trial.suggest_categorical("device_type", ['gpu']),
            "objective": "regression",
            "metric": "l2",
            "n_estimators": trial.suggest_categorical("n_estimators", [2000, 4000]),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "num_leaves": trial.suggest_int("num_leaves", 20, 300, step=10),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 20),
            "max_bin": trial.suggest_int("max_bin", 200, 300),
            "lambda_l1": trial.suggest_float("lambda_l1", 0, 5, step=0.2),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
            "bagging_fraction": trial.suggest_float(
                "bagging_fraction", 0.2, 0.9, step=0.1
            ),
            "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
            "feature_fraction": trial.suggest_float(
                "feature_fraction", 0.2, 0.9, step=0.1
            ),
        }

        cv = KFold(n_splits=5, shuffle=True, random_state=1121218)

        cv_scores = np.empty(5)
        for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = lgb.LGBMRegressor(**param_grid)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_test, y_test)],
                eval_metric="l2",
                callbacks=[
                    early_stopping(100),
                    log_evaluation(100)
                ],
            )
            preds = model.predict(X_test)
            cv_scores[idx] = mean_squared_error(y_test.values, preds)

        return np.mean(cv_scores)

    def train_ensemble(self, dataset: Union[str, 'list[tuple]'], splits: int = 5, use_data_augmentation=True):
        raise NotImplementedError("LGBM predictor doesn't support ensemble")

    def train(self, dataset: Union[str, 'list[Tuple]'], use_data_augmentation=True):
        # TODO
        if not isinstance(dataset, str):
            raise TypeError('LGBM supports only files, conversion to file is a TODO...')

        dataset_df = pd.read_csv(dataset)
        actual_b = self.__get_max_b(dataset_df)
        if self.feature_names is None:
            self.__setup_features_data(dataset_df)

        dataset_df = dataset_df.drop(columns=self.drop_feature_names)

        x = dataset_df.drop(columns=[self.label_name])
        y = dataset_df[self.label_name]

        train_ds = lgb.Dataset(x, y)
        train_log_folder = os.path.join(self.log_folder, f'B{actual_b}')
        create_empty_folder(train_log_folder)

        # train the model with random search
        if self.use_random_search:
            study = optuna.create_study(direction="minimize", study_name="LGBM Regressor")
            study.optimize(lambda trial: self.__objective(trial, x, y), n_trials=20)

            # LightGBMTunerCV(params, train_ds, folds=KFold(n_splits=5), callbacks=[early_stopping(100), log_evaluation(100)], return_cvbooster=True)

            self._logger.info("Best value (mse): %0.3f", study.best_value)
            self._logger.info("\tBest params:")

            for key, value in study.best_params.items():
                self._logger.info(f"\t\t{key}: {value}")

            self._logger.info("Training model with all samples, with best parameters found")
            self.model = lgb.train(study.best_params, train_ds)
        # else simply train the model with default parameters
        else:
            self.model = lgb.train({'objective': 'regression'}, train_ds)
            self._logger.info('LGBM training complete')

        if self.perform_feature_analysis:
            fig, ax = plt.subplots()
            lgb.plot_importance(self.model, ax)
            fig.savefig(os.path.join(train_log_folder, 'feature_importance.png'), bbox_inches='tight')

    def predict(self, x: list) -> float:
        # make sure categorical features are integers (and not float, pandas converts them to floats like 1.0)
        if self.cat_feature_indexes is not None:
            for cat_index in self.cat_feature_indexes:
                x[cat_index] = int(x[cat_index])

        # returned as a numpy array of single element
        return self.model.predict([x])[0]

    def predict_batch(self, x: 'Sequence[list]') -> np.ndarray:
        # make sure categorical features are integers (and not float, pandas converts them to floats like 1.0)
        if self.cat_feature_indexes is not None:
            for sample in x:
                for cat_index in self.cat_feature_indexes:
                    sample[cat_index] = int(sample[cat_index])

        # returned as a numpy array of single element
        preds = self.model.predict(x)  # type: np.ndarray
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

            real_values.append(predictions_df[self.label_name].values.tolist())
            predictions_df = predictions_df.drop(columns=[self.label_name] + self.drop_feature_names)
            prediction_samples.append(predictions_df.values.tolist())

        return dataset_paths, prediction_samples, real_values
