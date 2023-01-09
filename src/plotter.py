import logging
import os.path
import statistics

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score

import log_service
from utils.feature_utils import metrics_fields_dict
from utils.func_utils import compute_spearman_rank_correlation_coefficient_from_df, parse_cell_structures, compute_mape, intersection
from utils.plotter_utils import plot_histogram, plot_multibar_histogram, plot_boxplot, plot_pie_chart, plot_scatter, \
    plot_squared_scatter_chart, plot_pareto_front, plot_predictions_pareto_scatter_chart, generate_avg_max_min_bars

# Provides utility functions for plotting relevant data gained during the algorithm run,
# so that it can be further analyzed in a more straightforward way

__logger = None  # type: logging.Logger
# disable matplotlib info and warning messages
# TODO: it is not wise to disable warnings, but i couldn't find a way to only remove PostScript transparency warning,
#  which is a known thing and it is triggered thousands of times... EPS disabled, reverted to WARNING
plt.set_loglevel('WARNING')
# warnings.filterwarnings('ignore', module='matplotlib.backends.backend_ps')    # NOT WORKING

# Pareto objectives which require a predictor for estimation
predictor_objectives = ['time', 'accuracy', 'f1_score']


# TODO: otherwise would be initialized before run.py code, producing an error. Is there a less 'hacky' way?
def initialize_logger():
    global __logger
    __logger = log_service.get_logger(__name__)


def plot_specular_monoblock_info():
    __logger.info("Analyzing training_results.csv...")
    csv_path = log_service.build_path('csv', 'training_results.csv')
    df = pd.read_csv(csv_path)

    # take only mono block cells
    df = df[df['# blocks'] == 1]

    cells = parse_cell_structures(df['cell structure'])
    # cells have a single block, extrapolate the tuple instead of using the list of blocks
    first_block_iter = map(lambda blocks: blocks[0], cells)

    # unpack values into separate columns
    df['in1'], df['op1'], df['in2'], df['op2'] = zip(*first_block_iter)
    # filter only SMB
    smb_df = df[(df['in1'] == df['in2']) & (df['op1'] == df['op2']) & (df['in1'] == -1)]

    x = smb_df['op1']

    __logger.info("Writing plots...")
    plot_histogram(x, smb_df['training time(seconds)'], 'Operator', 'Time(s)', 'SMB (-1 input) training time', 'SMB_time', incline_labels=True)
    plot_histogram(x, smb_df['best val accuracy'], 'Operator', 'Val Accuracy', 'SMB (-1 input) validation accuracy', 'SMB_acc', incline_labels=True)
    plot_histogram(x, smb_df['val F1 score'], 'Operator', 'F1 score', 'SMB (-1 input) validation F1 score', 'SMB_F1', incline_labels=True)
    plot_histogram(x, smb_df['total params'], 'Operator', 'Params', 'SMB (-1 input) total parameters', 'SMB_params', incline_labels=True)
    plot_histogram(x, smb_df['flops'], 'Operator', 'FLOPS', 'SMB (-1 input) FLOPS', 'SMB_flops', incline_labels=True)
    __logger.info("SMB plots written successfully")


def plot_summary_training_info_per_block():
    __logger.info("Analyzing training overview data...")
    csv_path = log_service.build_path('csv', 'training_overview.csv')
    df = pd.read_csv(csv_path)

    x = df['# blocks']

    time_bars = generate_avg_max_min_bars(df['avg training time(s)'], df['max time'], df['min time'])
    score_bars = generate_avg_max_min_bars(df['avg val score'], df['max score'], df['min score'])

    plot_multibar_histogram(x, time_bars, 0.15, 'Blocks', 'Time(s)', 'Training time overview', 'train_time_overview')
    plot_multibar_histogram(x, score_bars, 0.15, 'Blocks', 'Score', 'Validation score overview', 'train_score_overview')

    __logger.info("Training aggregated overview plots written successfully")


def plot_cnn_train_boxplots_per_block(B: int):
    __logger.info("Analyzing training results data...")
    csv_path = log_service.build_path('csv', 'training_results.csv')
    df = pd.read_csv(csv_path)

    times_per_block, acc_per_block, f1_scores_per_block = [], [], []
    x = list(range(1, B + 1))
    for b in x:
        b_df = df[df['# blocks'] == b]

        acc_per_block.append(b_df[metrics_fields_dict['accuracy'].real_column])
        times_per_block.append(b_df[metrics_fields_dict['time'].real_column])
        f1_scores_per_block.append(b_df[metrics_fields_dict['f1_score'].real_column])

    plot_boxplot(acc_per_block, x, 'Blocks', 'Val accuracy', 'Val accuracy overview', 'val_acc_boxplot')
    plot_boxplot(times_per_block, x, 'Blocks', 'Training time', 'Training time overview', 'train_time_boxplot')
    plot_boxplot(f1_scores_per_block, x, 'Blocks', 'Val F1 score', 'Val F1 score overview', 'val_f1_score_boxplot')


def __initialize_dict_usage_data(keys: list):
    '''
    Create dictionary with indexes and initialize counters.
    '''
    counters_dict = {}

    for key in keys:
        counters_dict[key] = 0

    return counters_dict


def __update_counters(cells, op_counters: dict, input_counters: dict):
    '''
    Iterate cell structures and increment operators and inputs counters when the values are encountered.
    '''
    # iterate all cells (models selected for training)
    for cell in cells:
        # iterate on blocks tuple
        for in1, op1, in2, op2 in cell:
            op_counters[op1] += 1
            op_counters[op2] += 1
            input_counters[int(in1)] += 1
            input_counters[int(in2)] += 1

    return op_counters, input_counters


def __generate_value_list_from_op_counters_dict(op_counters: 'dict[str, int]', operations: 'list[str]'):
    return [op_counters[op] for op in operations]


def __generate_value_list_from_inputs_counters_dict(input_counters: 'dict[int, int]', inputs: 'list[int]'):
    return [input_counters[inp] for inp in inputs]


def plot_pareto_inputs_and_operators_usage(b: int, operators: 'list[str]', inputs: 'list[int]', limit: int = None):
    op_counters = __initialize_dict_usage_data(operators)
    input_counters = __initialize_dict_usage_data(inputs)

    __logger.info("Analyzing operators and inputs usage of pareto front for b=%d", b)
    csv_path = log_service.build_path('csv', f'pareto_front_B{b}.csv')
    df = pd.read_csv(csv_path)

    # used to limit the pareto front to the actual K children trained, if provided
    if limit:
        df = df.head(limit)

    cells = parse_cell_structures(df['cell structure'])

    op_counters, input_counters = __update_counters(cells, op_counters, input_counters)
    op_values = __generate_value_list_from_op_counters_dict(op_counters, operators)
    input_values = __generate_value_list_from_inputs_counters_dict(input_counters, inputs)

    plot_pie_chart(op_values, operators, f'Operators usage in b={b} pareto front', f'pareto_op_usage_B{b}')
    __logger.info("Pareto operators usage plot for b=%d written successfully", b)
    plot_pie_chart(input_values, inputs, f'Inputs usage in b={b} pareto front', f'pareto_inputs_usage_B{b}')
    __logger.info("Pareto inputs usage plot for b=%d written successfully", b)


# TODO: remove duplication with function above
def plot_exploration_inputs_and_operators_usage(b: int, operators: 'list[str]', inputs: 'list[int]'):
    op_counters = __initialize_dict_usage_data(operators)
    input_counters = __initialize_dict_usage_data(inputs)

    __logger.info("Analyzing operators and inputs usage of exploration pareto front for b=%d", b)
    csv_path = log_service.build_path('csv', f'exploration_pareto_front_B{b}.csv')

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        __logger.info('No exploration data found, skip plot generation')
        return

    if len(df) == 0:
        __logger.info('Exploration Pareto front was empty, skipping plot generation')
        return

    cells = parse_cell_structures(df['cell structure'])

    op_counters, input_counters = __update_counters(cells, op_counters, input_counters)
    op_values = __generate_value_list_from_op_counters_dict(op_counters, operators)
    input_values = __generate_value_list_from_inputs_counters_dict(input_counters, inputs)

    plot_pie_chart(op_values, operators, f'Operators usage in b={b} exploration pareto front', f'exploration_op_usage_B{b}')
    __logger.info("Pareto operators usage plot for b=%d written successfully", b)
    plot_pie_chart(input_values, inputs, f'Inputs usage in b={b} exploration pareto front', f'exploration_inputs_usage_B{b}')
    __logger.info("Pareto inputs usage plot for b=%d written successfully", b)


def plot_children_inputs_and_operators_usage(b: int, operators: 'list[str]', inputs: 'list[int]', children_cnn: 'list[str]'):
    op_counters = __initialize_dict_usage_data(operators)
    input_counters = __initialize_dict_usage_data(inputs)

    __logger.info("Analyzing operators and inputs usage of CNN children to train for b=%d", b)
    op_counters, input_counters = __update_counters(children_cnn, op_counters, input_counters)
    op_values = __generate_value_list_from_op_counters_dict(op_counters, operators)
    input_values = __generate_value_list_from_inputs_counters_dict(input_counters, inputs)

    plot_pie_chart(op_values, operators, f'Operations usage in b={b} CNN children', f'children_op_usage_B{b}')
    __logger.info("Children operators usage plot for b=%d written successfully", b)
    plot_pie_chart(input_values, inputs, f'Inputs usage in b={b} CNN children', f'children_inputs_usage_B{b}')
    __logger.info("Children inputs usage plot for b=%d written successfully", b)


def __build_prediction_dataframe(b: int, k: int, pnas_mode: bool):
    # PNAS mode has no pareto front, use sorted predictions (by score)
    if pnas_mode:
        pred_csv_path = log_service.build_path('csv', f'predictions_B{b}.csv')
        # PNAS takes best k cells, the others are useless
        pred_df = pd.read_csv(pred_csv_path).head(k)
    else:
        pareto_csv_path = log_service.build_path('csv', f'pareto_front_B{b}.csv')
        exploration_csv_path = log_service.build_path('csv', f'exploration_pareto_front_B{b}.csv')

        # POPNAS trains the top k cells of the pareto front, plus all the ones of the exploration front, if produced
        pred_df = pd.read_csv(pareto_csv_path).head(k)
        if os.path.exists(exploration_csv_path):
            exploration_df = pd.read_csv(exploration_csv_path)
            pred_df = pd.concat([pred_df, exploration_df], ignore_index=True)

    training_csv_path = log_service.build_path('csv', 'training_results.csv')
    training_df = pd.read_csv(training_csv_path)

    # take only trained CNN with correct block length
    training_df = training_df[training_df['# blocks'] == b]

    # now both dataframes have same length and they have the same CNN order. Confront items in order to get differences.
    return pd.merge(training_df, pred_df, on=['cell structure'], how='inner')


def plot_predictions_error(B: int, K: int, pnas_mode: bool, pareto_objectives: 'list[str]'):
    # build dataframes
    merge_dfs = [__build_prediction_dataframe(b, K, pnas_mode) for b in range(2, B + 1)]
    
    def plot_prediction_errors_for_metric(metric_name: str):
        '''
        Plot predictors related graphs, comparing the results with real values obtained after training.
         
        Args:
            metric_name: Pareto metric which has an associated predictor
        '''
        errors, avg_errors, max_errors, min_errors = [], np.zeros(B - 1), np.zeros(B - 1), np.zeros(B - 1)
        mapes, spearman_coeffs, r2_coeffs = np.zeros(B - 1), np.zeros(B - 1), np.zeros(B - 1)
        
        pred_values, real_values = [], []
        legend_labels = []

        m = metrics_fields_dict[metric_name]

        for b in range(2, B + 1):
            __logger.info("Comparing predicted values of %s with actual values retrieved after CNN training (b=%d)", metric_name, b)
            df = merge_dfs[b - 2]

            pred_b = df[m.pred_column]
            real_b = df[m.real_column]

            pred_values.append(pred_b.to_list())
            real_values.append(real_b.to_list())

            errors_b = real_b - pred_b
            errors.append(errors_b)

            avg_errors[b - 2] = statistics.mean(errors_b)
            max_errors[b - 2] = max(errors_b)
            min_errors[b - 2] = min(errors_b)

            mapes[b - 2] = compute_mape(real_b.to_list(), pred_b.to_list())
            spearman_coeffs[b - 2] = compute_spearman_rank_correlation_coefficient_from_df(df, m.real_column, m.pred_column)
            r2_coeffs[b - 2] = r2_score(real_b.to_list(), pred_b.to_list())

            legend_labels.append(f'B{b} (MAPE: {mapes[b - 2]:.3f}%, ρ: {spearman_coeffs[b - 2]:.3f})')

        x = np.arange(2, B + 1)
        bars = generate_avg_max_min_bars(avg_errors, max_errors, min_errors)
        capitalized_metric = metric_name.capitalize()
        units_str = f'({m.units})' if m.units is not None else ''

        plot_multibar_histogram(x, bars, 0.15, 'Blocks', f'{capitalized_metric}{units_str}',
                                  f'{capitalized_metric} prediction errors overview (real - predicted)', f'pred_{metric_name}_errors_overview')
        plot_boxplot(errors, x, 'Blocks', f'{capitalized_metric} error{units_str}',
                       f'{capitalized_metric} prediction errors overview (real - predicted)', f'pred_{metric_name}_errors_boxplot')
        plot_squared_scatter_chart(real_values, pred_values, f'Real {metric_name}{units_str}', f'Predicted {metric_name}{units_str}',
                                     f'{capitalized_metric} predictions overview', f'{metric_name}_pred_overview', legend_labels=legend_labels)

    # write plots about each Pareto metric associated to a predictor
    for p_obj in intersection(pareto_objectives, predictor_objectives):
        plot_prediction_errors_for_metric(p_obj)

    __logger.info("Prediction error overview plots written successfully")


def plot_pareto_front_curves(B: int, pareto_objectives: 'list[str]'):
    training_csv_path = log_service.build_path('csv', 'training_results.csv')
    training_df = pd.read_csv(training_csv_path)
    training_df = training_df[training_df['exploration'] == False]

    for b in range(2, B + 1):
        # front built with actual values got from training
        training_df_b = training_df[training_df['# blocks'] == b]

        # pareto front predicted
        pareto_b_csv_path = log_service.build_path('csv', f'pareto_front_B{b}.csv')
        pareto_df = pd.read_csv(pareto_b_csv_path)

        real_coords, pred_coords = [], []

        for objective in pareto_objectives:
            m = metrics_fields_dict[objective]
            real_coords.append(training_df_b[m.real_column].to_list())
            pred_coords.append(pareto_df[m.pred_column].to_list())

        plot_pareto_front(real_coords, pred_coords, pareto_objectives,
                          title=f'3D Pareto front B{b}', save_name=f'pareto_plot_B{b}_3D')

    __logger.info("Pareto front curve plots written successfully")


def plot_predictions_with_pareto_analysis(B: int, pareto_objectives: 'list[str]'):
    for b in range(2, B + 1):
        predictions_df = pd.read_csv(log_service.build_path('csv', f'predictions_B{b}.csv'))
        pareto_df = pd.read_csv(log_service.build_path('csv', f'pareto_front_B{b}.csv'))

        pred_coords, pareto_coords = [], []

        for objective in pareto_objectives:
            m = metrics_fields_dict[objective]
            pred_coords.append(predictions_df[m.pred_column].to_list())
            pareto_coords.append(pareto_df[m.pred_column].to_list())

        plot_predictions_pareto_scatter_chart(pred_coords, pareto_coords, pareto_objectives,
                                                f'Predictions with Pareto points B{b}', f'predictions_with_pareto_B{b}')

    __logger.info("Predictions-Pareto analysis plots written successfully")


def plot_multi_output_boxplot():
    __logger.info("Analyzing accuracy of the multiple outputs for each model...")
    csv_path = log_service.build_path('csv', 'multi_output.csv')
    outputs_df = pd.read_csv(csv_path).drop(columns=['cell_spec'])

    col_names = outputs_df.columns.to_list()    # type: list[str]
    accuracy_df = outputs_df[[col for col in col_names if col.endswith('accuracy')]]
    f1_df = outputs_df[[col for col in col_names if col.endswith('f1_score')]]

    def plot_multi_output_metric_boxplots(metric_df: pd.DataFrame, metric_name: str):
        # remove all rows that have some missing values (this means the cell don't use -1 lookback, stacking less cells)
        # removing these rows allow a fair comparison in the plot
        lb1_df = metric_df.dropna()
        lb1_x_labels = lb1_df.columns.to_list()
        lb1_series_list = [lb1_df[col] for col in lb1_x_labels]

        # get the entries with less cells (use only -2 lookback)
        # TODO: -2 because it's the lookback used, expand the logic to other lookbacks if needed
        lb2_df = metric_df[metric_df[lb1_x_labels[-2]].isnull()]
        lb2_x_labels = lb1_x_labels[::-2][::-1]
        lb2_series_list = [lb2_df[output_key] for output_key in lb2_x_labels]

        # get only the cell index
        lb1_x_labels = [str(label).split('_')[0] for label in lb1_x_labels]
        lb2_x_labels = lb1_x_labels[::-2][::-1]

        plot_boxplot(lb1_series_list, lb1_x_labels, 'Cell outputs', metric_name, f'Best {metric_name} per output', f'multi_output_{metric_name}_lb1')
        plot_boxplot(lb2_series_list, lb2_x_labels, 'Cell outputs', metric_name,
                       f'Best {metric_name} per output (-2 lookback)', f'multi_output_{metric_name}_lb2')

    plot_multi_output_metric_boxplots(accuracy_df, 'accuracy')
    plot_multi_output_metric_boxplots(f1_df, 'f1_score')

    __logger.info("Multi output overview plot written successfully")


def plot_correlations_with_training_time():
    __logger.info('Writing time correlation plots...')

    training_csv_path = log_service.build_path('csv', 'training_results.csv')
    training_df = pd.read_csv(training_csv_path)

    training_times = training_df['training time(seconds)'].to_list()
    params = training_df['total params'].to_list()
    flops = training_df['flops'].to_list()
    inference_times = training_df['inference time(seconds)'].to_list()

    def _compute_correlations_and_plot(metric_list: list, train_times: list, metric_name: str):
        pearson, _ = pearsonr(metric_list, train_times)
        spearman, _ = spearmanr(metric_list, train_times)    # types are correct, they are wrongly declared in scipy
        labels = [f'Pearson: {pearson:.3f}, Spearman: {spearman:.3f}']

        plot_scatter(metric_list, train_times, metric_name, 'time(s)',
                       f'({metric_name.capitalize()}, Training time) correlation', f'{metric_name.replace(" ", "_")}_time_corr', legend_labels=labels)

    _compute_correlations_and_plot(params, training_times, metric_name='params')
    _compute_correlations_and_plot(flops, training_times, metric_name='flops')
    _compute_correlations_and_plot(inference_times, training_times, metric_name='inference time')

    __logger.info('Time correlation plots written successfully')
