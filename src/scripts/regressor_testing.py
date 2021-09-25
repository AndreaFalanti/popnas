import argparse
import logging
import os
import shutil
import sys
from configparser import ConfigParser
from contextlib import redirect_stdout, redirect_stderr

import catboost
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..utils.stream_to_logger import StreamToLogger
from ..utils.catboost_eval_metric_spearman import CatBoostEvalMetricSpearman

# TODO: deleting this would cause import failures inside aMLLibrary files, but from POPNAS its better to
# import them directly to enable intellisense
amllibrary_path = os.path.join(os.getcwd(), 'src', 'aMLLibrary')
sys.path.append(amllibrary_path)

from ..aMLLibrary import sequence_data_processing


def setup_folders(log_path, techniques):
    regressors_test_path = os.path.join(log_path, 'regressors_test')
    try:
        os.makedirs(regressors_test_path)
    except OSError:
        shutil.rmtree(regressors_test_path)
        os.makedirs(regressors_test_path)

    for technique in techniques:
        os.makedirs(os.path.join(regressors_test_path, technique))

    return regressors_test_path


def create_logger(name, log_path):
    logger = logging.getLogger(name)

    # Create handlers
    file_handler = logging.FileHandler(os.path.join(log_path, 'debug.log'))
    file_handler.setFormatter(logging.Formatter("%(asctime)s - [%(name)s:%(levelname)s] %(message)s"))
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])

    return logger


def get_feature_names(training_time_df: pd.DataFrame):
    training_time_df = training_time_df.drop(columns=['time'])
    return training_time_df.columns.values.tolist()


def prepare_regressor_data(training_time_df: pd.DataFrame, log_path, b: int):
    '''
    Write training csv with only CNN up to b blocks.
    Returns csv path and predictions to made after regressor have been trained.

    Returns:
    (str): csv path
    (list[str]): regressor features of models to predict (models with b+1 blocks)
    (list[float]): real times of the models to predict
    '''
    iter_df = training_time_df[training_time_df['blocks'] <= b]
    predictions_df = training_time_df[training_time_df['blocks'] == (b + 1)]

    save_path = os.path.join(log_path, f'inputs_B{b}.csv')
    iter_df.to_csv(save_path, index=False)

    real_times = predictions_df['time'].values.tolist()
    predictions_df = predictions_df.drop(columns=['time'])
    models_to_predict = predictions_df.values.tolist()

    return save_path, models_to_predict, real_times


def write_regressor_config_file(input_csv_path, log_path, techniques: 'list[str]', actual_b: int):
    '''
    Note: technique is a list of single element (taken from controller, use list type to avoid to refactor)
    '''
    config = ConfigParser()
    # to keep casing in keys while reading / writing
    config.optionxform = str

    config.read(os.path.join('src', 'configs', 'regressors.ini'))

    for section in config.sections():
        if section == 'General':
            continue

        # delete config section not relevant to selected techniques
        if section not in techniques:
            del config[section]

    # value in .ini must be a single string of format ['technique1', 'technique2', ...]
    # note: '' are important for correct execution (see map)
    techniques_iter = map(lambda s: f"'{s}'", techniques)
    techniques_str = f"[{', '.join(techniques_iter)}]"
    config['General']['techniques'] = techniques_str
    config['DataPreparation'] = {'input_path': input_csv_path}

    save_path = os.path.join(log_path, f'B{actual_b}.ini')
    with open(save_path, 'w') as f:
        config.write(f)

    return save_path


def build_best_regressor(config_path, log_path, logger, b: int):
    '''
    Inside the config it's specified the technique to apply, the configurations to test
    and the input file to use.
    '''

    save_path = os.path.join(log_path, f'output_B{b}')
    # a-MLLibrary, redirect output to script logger (it uses stderr for output, see custom logger)
    redir_logger = StreamToLogger(logger)

    with redirect_stdout(redir_logger):
        with redirect_stderr(redir_logger):
            sequence_data_processor = sequence_data_processing.SequenceDataProcessing(config_path, output=save_path)
            best_regressor = sequence_data_processor.process()

    return best_regressor


def build_catboost_model(input_data_path: str, col_desc_path: str, logger: logging.Logger):
    train_pool = catboost.Pool(input_data_path, delimiter=',', has_header=True, column_description=col_desc_path)

    # param_grid = {
    #     'learning_rate': [0.1, 0.15],
    #     'depth': [4, 5, 6, 7],
    #     'l2_leaf_reg': [1, 3, 5, 7],
    #     'random_strength': [1, 1.25, 1.4],
    #     'bagging_temperature': [0.6, 0.75, 1],
    #     'grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide']
    # }
    param_grid = {
        'depth': [6],
        'l2_leaf_reg': [1],
        'random_strength': [1.25]
    }

    redir_logger = StreamToLogger(logger)
    with redirect_stdout(redir_logger):
        with redirect_stderr(redir_logger):
            # specify the training parameters
            # TODO: task type = 'GPU' is very slow, why?
            model = catboost.CatBoostRegressor(custom_metric='MAPE', eval_metric=CatBoostEvalMetricSpearman(), early_stopping_rounds=16)
            # train the model
            results_dict = model.grid_search(param_grid, train_pool, train_size=0.85)

    logger.info('Best parameters: %s', str(results_dict['params']))
    return model


def initialize_scatter_plot_dict(techniques: 'list[str]') -> 'dict[str, dict]':
    # dictionary of dictionaries to store values for each regressor technique
    # each technique will have x and y fields, that are list of lists
    # and MAPE and spearman fields, which are plain lists (of len = B)
    scatter_values = {}
    for technique in techniques:
        scatter_values[technique] = {}
        scatter_values[technique]['x'] = []
        scatter_values[technique]['y'] = []
        scatter_values[technique]['MAPE'] = []
        scatter_values[technique]['spearman'] = []

    return scatter_values


def plot_squared_scatter_chart(x, y, technique, log_path, plot_reference=True, legend_labels=None):
    fig, ax = plt.subplots()

    # list of lists with same dimensions are required, or also flat lists with same dimensions
    assert len(x) == len(y)

    # list of lists case
    if any(isinstance(el, list) for el in x):
        assert len(x) == len(legend_labels)

        colors = cm.rainbow(np.linspace(0, 1, len(x)))
        for xs, ys, color, lab in zip(x, y, colors, legend_labels):
            plt.scatter(xs, ys, color=color, label=lab)
    else:
        plt.scatter(x, y)

    plt.xlabel('Real time(seconds)')
    plt.ylabel('Predicted time(seconds)')
    plt.title(f'Time predictions overview ({technique})')
    plt.gca().set_aspect('equal', adjustable='box')

    plt.legend(fontsize='x-small')

    # add reference line (bisector line x = y)
    if plot_reference:
        ax_lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]

        ax.plot(ax_lims, ax_lims, '--k', alpha=0.75)

    log_path = os.path.join(log_path, 'results.png')
    plt.savefig(log_path, bbox_inches='tight')


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-p', metavar='FOLDER', type=str, help="log folder", required=True)
    args = parser.parse_args()

    # aMLLibrary techniques to test
    regressor_techniques = ['CatBoost', 'SVR', 'NNLS', 'XGBoost', 'LRRidge']

    csv_path = os.path.join(args.p, 'csv')
    log_path = setup_folders(args.p, techniques=regressor_techniques)

    logger = create_logger(__name__, log_path)

    training_time_df = pd.read_csv(os.path.join(csv_path, 'training_time.csv'))

    max_b = training_time_df['blocks'].max()
    scatter_plot_legends = []
    scatter_values = initialize_scatter_plot_dict(regressor_techniques)

    feature_columns = get_feature_names(training_time_df)
    catboost_col_desc_file_path = os.path.join(csv_path, 'column_desc.csv')

    # regressor is called after b=1 training and above, but not at last step (max_b)
    for b in range(1, max_b):
        # label of predictions
        scatter_plot_legends.append(f'B{b + 1}')

        input_csv_path, models_to_predict, prediction_real_times = prepare_regressor_data(training_time_df, log_path, b)
        logger.info('Built regressor input for b=%d', b)

        for technique in regressor_techniques:
            logger.info(f'------------------- {technique} B={b} -----------------------')
            technique_log_path = os.path.join(log_path, technique)

            if technique == 'CatBoost':
                best_regressor = build_catboost_model(input_csv_path, catboost_col_desc_file_path, logger)
            else:
                config_path = write_regressor_config_file(input_csv_path, technique_log_path, [technique], b)
                best_regressor = build_best_regressor(config_path, technique_log_path, logger, b)

            scatter_x, scatter_y = [], []
            for model_pred_features, real_time in zip(models_to_predict, prediction_real_times):
                if technique == 'CatBoost':
                    cat_features_indexes = [indexes for i in range(max_b) for indexes in (4*i+1, 4*i+3)]
                    # make sure categorical features are integers (and not float, pandas converts them to floats like 1.0)
                    for cat_index in cat_features_indexes:
                        model_pred_features[cat_index] = int(model_pred_features[cat_index])

                    features_pool = catboost.Pool([model_pred_features], cat_features=cat_features_indexes, feature_names=feature_columns)
                    # returned as a numpy array of single element
                    predicted_time = best_regressor.predict(features_pool)[0]
                else:
                    features_df = pd.DataFrame([model_pred_features], columns=feature_columns)
                    predicted_time = best_regressor.predict(features_df)[0]

                scatter_x.append(real_time)
                scatter_y.append(predicted_time)

            comparison_df = pd.DataFrame({'real_time': scatter_x, 'pred_time': scatter_y})
            scatter_values[technique]['x'].append(scatter_x)
            scatter_values[technique]['y'].append(scatter_y)
            scatter_values[technique]['MAPE'].append((((comparison_df['real_time'] - comparison_df['pred_time']) /
                                                       comparison_df['real_time']).abs()).mean() * 100)
            scatter_values[technique]['spearman'].append(
                comparison_df['real_time'].corr(comparison_df['pred_time'], method='spearman'))

            logger.info('--------------------------------------------------------------')

    logger.info('Built plots for each regressor')
    for technique in regressor_techniques:
        technique_log_path = os.path.join(log_path, technique)
        # add MAPE and spearman to legend
        technique_legend_labels = list(
            map(lambda label, mape, spearman: label + f' (MAPE: {mape:.3f}%, ρ: {spearman:.3f}',
                scatter_plot_legends, scatter_values[technique]['MAPE'], scatter_values[technique]['spearman']))

        plot_squared_scatter_chart(scatter_values[technique]['x'], scatter_values[technique]['y'], technique,
                                   technique_log_path, legend_labels=technique_legend_labels)

    logger.info('Script completed successfully')


if __name__ == '__main__':
    main()
