import logging
import os.path
import statistics

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score

import log_service
from models.results.base import TargetMetric
from search_space import CellSpecification, parse_cell_strings
from utils.cell_counter import CellCounter
from utils.func_utils import compute_spearman_rank_correlation_coefficient_from_df, compute_mape
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


# TODO: otherwise would be initialized before run.py code, producing an error. Is there a less 'hacky' way?
def initialize_logger():
    global __logger
    __logger = log_service.get_logger(__name__)


def plot_specular_monoblock_info(target_metrics: 'list[TargetMetric]'):
    __logger.info("Analyzing training_results.csv...")
    csv_path = log_service.build_path('csv', 'training_results.csv')
    df = pd.read_csv(csv_path)

    # take only mono block cells
    df = df[df['# blocks'] == 1]

    cells = parse_cell_strings(df['cell structure'])
    # cells have a single block, extrapolate the tuple instead of using the list of blocks
    first_block_iter = map(lambda blocks: blocks[0], cells)

    # unpack values into separate columns
    df['in1'], df['op1'], df['in2'], df['op2'] = zip(*first_block_iter)
    # filter only SMB
    smb_df = df[(df['in1'] == df['in2']) & (df['op1'] == df['op2']) & (df['in1'] == -1)]

    # use as labels the first operator (that is actually the same of the second one)
    x = smb_df['op1']
    x_axis_label = 'Operator'

    __logger.info("Writing plots...")
    for m in target_metrics:
        y = smb_df[m.results_csv_column]
        title = f'SMB (-1 input) {m.name}'
        plot_histogram(x, y, x_axis_label, m.plot_label(), title, save_name=f'SMB_{m.name}', incline_labels=True)

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


def plot_metrics_boxplot_per_block(B: int, target_metrics: 'list[TargetMetric]'):
    __logger.info("Analyzing training results data...")
    csv_path = log_service.build_path('csv', 'training_results.csv')
    df = pd.read_csv(csv_path)

    # use block numbers as x labels, and get dataframes partitions for each block step.
    x = list(range(1, B + 1))
    x_axis_label = 'Blocks'
    block_dfs = [df[df['# blocks'] == b] for b in x]

    for m in target_metrics:
        data = [b_df[m.results_csv_column] for b_df in block_dfs]
        plot_boxplot(data, x, x_axis_label, m.name, title=f'{m.name} overview', save_name=f'{m.name}_boxplot')


def _get_cell_elems_usages_as_ordered_lists(cells: 'list[CellSpecification]', input_keys: 'list[int]', op_keys: 'list[str]'):
    cell_counter = CellCounter(input_keys, op_keys)
    for cell_spec in cells:
        cell_counter.update_from_cell_spec(cell_spec)

    return cell_counter.to_lists()


def plot_pareto_inputs_and_operators_usage(b: int, operators: 'list[str]', inputs: 'list[int]', limit: int = None):
    __logger.info("Analyzing operators and inputs usage of pareto front for b=%d", b)
    csv_path = log_service.build_path('csv', f'pareto_front_B{b}.csv')
    df = pd.read_csv(csv_path)

    # used to limit the pareto front to the actual K children trained, if provided
    if limit:
        df = df.head(limit)

    cells = parse_cell_strings(df['cell structure'])
    input_values, op_values = _get_cell_elems_usages_as_ordered_lists(cells, inputs, operators)

    plot_pie_chart(op_values, operators, f'Operators usage in b={b} pareto front', f'pareto_op_usage_B{b}')
    __logger.info("Pareto operators usage plot for b=%d written successfully", b)
    plot_pie_chart(input_values, inputs, f'Inputs usage in b={b} pareto front', f'pareto_inputs_usage_B{b}')
    __logger.info("Pareto inputs usage plot for b=%d written successfully", b)


def plot_exploration_inputs_and_operators_usage(b: int, operators: 'list[str]', inputs: 'list[int]'):
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

    cells = parse_cell_strings(df['cell structure'])
    input_values, op_values = _get_cell_elems_usages_as_ordered_lists(cells, inputs, operators)

    plot_pie_chart(op_values, operators, f'Operators usage in b={b} exploration pareto front', f'exploration_op_usage_B{b}')
    __logger.info("Exploration Pareto operators usage plot for b=%d written successfully", b)
    plot_pie_chart(input_values, inputs, f'Inputs usage in b={b} exploration pareto front', f'exploration_inputs_usage_B{b}')
    __logger.info("Exploration Pareto inputs usage plot for b=%d written successfully", b)


def plot_children_inputs_and_operators_usage(b: int, operators: 'list[str]', inputs: 'list[int]', children_cnn: 'list[list]'):
    __logger.info("Analyzing operators and inputs usage of CNN children to train for b=%d", b)
    input_values, op_values = _get_cell_elems_usages_as_ordered_lists(children_cnn, inputs, operators)

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


def plot_predictions_error(B: int, K: int, pnas_mode: bool, pareto_predictable_metrics: 'list[TargetMetric]'):
    x = np.arange(2, B + 1)
    # build dataframes
    merged_dfs = [__build_prediction_dataframe(b, K, pnas_mode) for b in x]
    
    def plot_prediction_errors_for_metric(metric: TargetMetric):
        '''
        Plot predictors related graphs, comparing the results with real values obtained after training.
         
        Args:
            metric: TargetMetric of a Pareto objective
        '''
        errors, avg_errors, max_errors, min_errors = [], np.zeros(B - 1), np.zeros(B - 1), np.zeros(B - 1)
        mapes, spearman_coeffs, r2_coeffs = np.zeros(B - 1), np.zeros(B - 1), np.zeros(B - 1)
        
        pred_values, real_values = [], []
        legend_labels = []

        for i, df in enumerate(merged_dfs):
            __logger.info("Comparing predicted values of %s with actual values retrieved after CNN training (b=%d)", metric.name, i + 2)

            pred_b = df[metric.prediction_csv_column]
            real_b = df[metric.results_csv_column]
            errors_b = real_b - pred_b

            pred_values.append(pred_b.to_list())
            real_values.append(real_b.to_list())
            errors.append(errors_b)

            avg_errors[i] = statistics.mean(errors_b)
            max_errors[i] = max(errors_b)
            min_errors[i] = min(errors_b)

            mapes[i] = compute_mape(real_b, pred_b)
            spearman_coeffs[i] = compute_spearman_rank_correlation_coefficient_from_df(real_b, pred_b)
            r2_coeffs[i] = r2_score(real_b, pred_b)

            legend_labels.append(f'B{i + 2} (MAPE: {mapes[i]:.3f}%, ρ: {spearman_coeffs[i]:.3f})')

        bars = generate_avg_max_min_bars(avg_errors, max_errors, min_errors)
        y_axis_label = metric.plot_label()

        plot_multibar_histogram(x, bars, 0.15, 'Blocks', y_axis_label,
                                  f'{metric.name} prediction errors overview (real - predicted)', f'pred_{metric.name}_errors_overview')
        plot_boxplot(errors, x, 'Blocks', f'{y_axis_label} error',
                       f'{metric.name} prediction errors overview (real - predicted)', f'pred_{metric.name}_errors_boxplot')
        plot_squared_scatter_chart(real_values, pred_values, f'Real {y_axis_label}', f'Predicted {y_axis_label}',
                                     f'{metric.name} predictions overview', f'{metric.name}_pred_overview', legend_labels=legend_labels)

    # write plots about each Pareto metric associated to a predictor
    for m in pareto_predictable_metrics:
        plot_prediction_errors_for_metric(m)

    __logger.info("Prediction error overview plots written successfully")


def plot_pareto_front_curves(B: int, pareto_metrics: 'list[TargetMetric]'):
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

        for m in pareto_metrics:
            real_coords.append(training_df_b[m.results_csv_column])
            pred_coords.append(pareto_df[m.prediction_csv_column])

        pareto_objective_names = [m.name for m in pareto_metrics]
        plot_pareto_front(real_coords, pred_coords, pareto_objective_names,
                          title=f'3D Pareto front B{b}', save_name=f'pareto_plot_B{b}_3D')

    __logger.info("Pareto front curve plots written successfully")


def plot_predictions_with_pareto_analysis(B: int, pareto_metrics: 'list[TargetMetric]'):
    for b in range(2, B + 1):
        predictions_df = pd.read_csv(log_service.build_path('csv', f'predictions_B{b}.csv'))
        pareto_df = pd.read_csv(log_service.build_path('csv', f'pareto_front_B{b}.csv'))

        pred_coords, pareto_coords = [], []

        for m in pareto_metrics:
            pred_coords.append(predictions_df[m.prediction_csv_column].to_list())
            pareto_coords.append(pareto_df[m.prediction_csv_column].to_list())

        pareto_objective_names = [m.name for m in pareto_metrics]
        plot_predictions_pareto_scatter_chart(pred_coords, pareto_coords, pareto_objective_names,
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
