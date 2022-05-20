import logging
import os.path
import statistics
from typing import NamedTuple, Optional

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.io.parsers import TextFileReader
from sklearn.metrics import r2_score

import log_service
from utils.func_utils import compute_spearman_rank_correlation_coefficient_from_df, parse_cell_structures, compute_mape

# Provides utility functions for plotting relevant data gained during the algorithm run,
# so that it can be further analyzed in a more straightforward way

__logger = None  # type: logging.Logger
# disable matplotlib info messages
plt.set_loglevel('WARNING')


def initialize_logger():
    global __logger
    __logger = log_service.get_logger(__name__)


def __save_and_close_plot(fig, save_name):
    # save as png
    save_path = log_service.build_path('plots', save_name + '.png')
    plt.savefig(save_path, bbox_inches='tight')

    plt.close(fig)


def __save_latex_plots(save_name):
    # save as eps (good format for Latex)
    save_path = log_service.build_path('plots', 'eps', save_name + '.eps')
    plt.savefig(save_path, bbox_inches='tight', format='eps')

    # save as pdf
    save_path = log_service.build_path('plots', 'pdf', save_name + '.pdf')
    plt.savefig(save_path, bbox_inches='tight')


def __plot_histogram(x, y, x_label, y_label, title, save_name, incline_labels=False):
    fig = plt.figure()
    plt.bar(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # add y grid lines
    plt.grid(b=True, which='both', axis='y', alpha=0.5, color='k')

    # use inclined x-labels
    if incline_labels:
        plt.gcf().autofmt_xdate()

    __save_latex_plots(save_name)
    plt.title(title)
    __save_and_close_plot(fig, save_name)


def __plot_multibar_histogram(x, y_array: 'list[BarInfo]', col_width, x_label, y_label, title, save_name):
    fig, ax = plt.subplots()

    x_label_dist = np.arange(len(x))  # the label locations
    x_offset = - ((len(y_array) - 1) / 2.0) * col_width

    # Make the plot
    for bar_info in y_array:
        ax.bar(x_label_dist + x_offset, bar_info.y, color=bar_info.color, width=col_width, label=bar_info.label)
        x_offset += col_width

    ax.set_xticks(x_label_dist)
    ax.set_xticklabels(x)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # add y grid lines
    plt.grid(b=True, which='both', axis='y', alpha=0.5, color='silver')

    ax.legend()

    __save_latex_plots(save_name)
    ax.set_title(title)
    __save_and_close_plot(fig, save_name)


def __plot_boxplot(values, labels, x_label, y_label, title, save_name, incline_labels=False):
    fig = plt.figure()
    plt.boxplot(values, labels=labels)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # add y grid lines
    plt.grid(b=True, which='both', axis='y', alpha=0.5, color='k')

    # use inclined x-labels
    if incline_labels:
        plt.gcf().autofmt_xdate()

    __save_latex_plots(save_name)
    plt.title(title)
    __save_and_close_plot(fig, save_name)


def __plot_pie_chart(labels, values, title, save_name):
    total = sum(values)

    fig, ax = plt.subplots()

    pie_cm = plt.get_cmap('tab20')
    colors = pie_cm(np.linspace(0, 1.0 / 20 * len(labels) - 0.01, len(labels)))

    explode = np.empty(len(labels))  # type: np.ndarray
    explode.fill(0.03)

    # label, percentage, value are written only in legend, to avoid overlapping texts in chart
    legend_labels = [f'{label} - {(val / total) * 100:.3f}% ({val:.0f})' for label, val in zip(labels, values)]

    patches, texts = ax.pie(values, labels=labels, explode=explode, startangle=90, labeldistance=None, colors=colors)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.legend(patches, legend_labels, loc='lower left', bbox_to_anchor=(1.03, 0.04))
    plt.subplots_adjust(right=0.7)

    __save_latex_plots(save_name)
    plt.title(title)
    __save_and_close_plot(fig, save_name)


def __plot_squared_scatter_chart(x, y, x_label, y_label, title, save_name,
                                 plot_reference: bool = True, legend_labels: Optional['list[str]'] = None, value_range: Optional[tuple] = None):
    fig, ax = plt.subplots()

    # list of lists with same dimensions are required, or also flat lists with same dimensions
    assert len(x) == len(y)

    # list of lists case
    if any(isinstance(el, list) for el in x):
        assert len(x) == len(legend_labels)

        colors = cm.rainbow(np.linspace(0, 1, len(x)))
        for xs, ys, color, lab in zip(x, y, colors, legend_labels):
            plt.scatter(xs, ys, marker='.', color=color, label=lab)
    else:
        plt.scatter(x, y)

    if value_range is not None:
        plt.xlim(*value_range)
        plt.ylim(*value_range)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.legend(fontsize='x-small')

    # add reference line (bisector line x = y)
    if plot_reference:
        ax_lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]

        ax.plot(ax_lims, ax_lims, '--k', alpha=0.75)

    __save_latex_plots(save_name)
    plt.title(title)
    __save_and_close_plot(fig, save_name)


def __plot_pareto_front(x_real: list, y_real: list, x_pred: list, y_pred: list, title: str, save_name: str):
    fig, ax = plt.subplots()

    # trans_offset = mtransforms.offset_copy(ax.transData, fig=fig, x=0, y=-0.10, units='inches')

    plt.plot(x_real, y_real, '--.b', x_pred, y_pred, '--.g', alpha=0.6)
    # for i, (x, y) in enumerate(zip(x_real, y_real)):
    #     plt.text(x, y, str(i), color='red', fontsize=12, transform=trans_offset)

    plt.xlabel('time')
    plt.ylabel('accuracy')

    __save_latex_plots(save_name)
    plt.title(title)
    __save_and_close_plot(fig, save_name)


def __plot_3d_pareto_front(x_real: list, y_real: list, x_pred: list, y_pred: list, title: str, save_name: str):
    # fig, ax = plt.subplots()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    x_real.reverse()
    y_real.reverse()
    x_pred.reverse()
    y_pred.reverse()

    ax.plot(list(range(len(x_real))), x_real, y_real, '--.b', alpha=0.6)
    ax.plot(list(range(len(x_pred))), x_pred, y_pred, '--.g', alpha=0.6)

    ax.set_xlabel('rank')
    ax.set_ylabel('time')
    ax.set_zlabel('accuracy')

    __save_latex_plots(save_name)
    plt.title(title)
    __save_and_close_plot(fig, save_name)


def __generate_avg_max_min_bars(avg_vals, max_vals, min_vals):
    '''
    Build avg, max and min bars for multi-bar plots.

    Args:
        avg_vals ([type]): values to assign to avg bar
        max_vals ([type]): values to assign to max bar
        min_vals ([type]): values to assign to min bar

    Returns:
        (list[BarInfo, BarInfo, BarInfo]): BarInfos usable in multi-bar plots
    '''
    bar_avg = BarInfo(avg_vals, 'b', 'avg')
    bar_max = BarInfo(max_vals, 'g', 'max')
    bar_min = BarInfo(min_vals, 'r', 'min')

    return [bar_avg, bar_max, bar_min]


def plot_dynamic_reindex_related_blocks_info():
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

    df = df[(df['in1'] == df['in2']) & (df['op1'] == df['op2']) & (df['in1'] == -1)]

    x = df['op1']
    y_time = df['training time(seconds)']
    y_acc = df['best val accuracy']
    y_params = df['total params']
    y_flops = df['flops']

    __logger.info("Writing plots...")
    __plot_histogram(x, y_time, 'Operation', 'Time(s)', 'SMB (-1 input) training time', 'SMB_time', incline_labels=True)
    __plot_histogram(x, y_acc, 'Operation', 'Val Accuracy', 'SMB (-1 input) validation accuracy', 'SMB_acc', incline_labels=True)
    __plot_histogram(x, y_params, 'Operation', 'Params', 'SMB (-1 input) total parameters', 'SMB_params', incline_labels=True)
    __plot_histogram(x, y_flops, 'Operation', 'FLOPS', 'SMB (-1 input) FLOPS', 'SMB_flops', incline_labels=True)
    __logger.info("SMB plots written successfully")


def plot_training_info_per_block():
    __logger.info("Analyzing training overview data...")
    csv_path = log_service.build_path('csv', 'training_overview.csv')
    df = pd.read_csv(csv_path)

    x = df['# blocks']

    time_bars = __generate_avg_max_min_bars(df['avg training time(s)'], df['max time'], df['min time'])
    acc_bars = __generate_avg_max_min_bars(df['avg val acc'], df['max acc'], df['min acc'])

    __plot_multibar_histogram(x, time_bars, 0.15, 'Blocks', 'Time(s)', 'Training time overview', 'train_time_overview')
    __plot_multibar_histogram(x, acc_bars, 0.15, 'Blocks', 'Accuracy', 'Validation accuracy overview', 'train_acc_overview')

    __logger.info("Training aggregated overview plots written successfully")


def plot_cnn_train_boxplots_per_block(B: int):
    __logger.info("Analyzing training results data...")
    csv_path = log_service.build_path('csv', 'training_results.csv')
    df = pd.read_csv(csv_path)

    times_per_block, acc_per_block = [], []
    x = list(range(1, B + 1))
    for b in x:
        b_df = df[df['# blocks'] == b]

        acc_per_block.append(b_df['best val accuracy'])
        times_per_block.append(b_df['training time(seconds)'])

    __plot_boxplot(acc_per_block, x, 'Blocks', 'Val accuracy', 'Val accuracy overview', 'val_acc_boxplot')
    __plot_boxplot(times_per_block, x, 'Blocks', 'Training time', 'Training time overview', 'train_time_boxplot')


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

    __plot_pie_chart(operators, op_values, f'Operators usage in b={b} pareto front', f'pareto_op_usage_B{b}')
    __logger.info("Pareto operators usage plot for b=%d written successfully", b)
    __plot_pie_chart(inputs, input_values, f'Inputs usage in b={b} pareto front', f'pareto_inputs_usage_B{b}')
    __logger.info("Pareto inputs usage plot for b=%d written successfully", b)


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

    cells = parse_cell_structures(df['cell structure'])

    op_counters, input_counters = __update_counters(cells, op_counters, input_counters)
    op_values = __generate_value_list_from_op_counters_dict(op_counters, operators)
    input_values = __generate_value_list_from_inputs_counters_dict(input_counters, inputs)

    __plot_pie_chart(operators, op_values, f'Operators usage in b={b} exploration pareto front', f'exploration_op_usage_B{b}')
    __logger.info("Pareto operators usage plot for b=%d written successfully", b)
    __plot_pie_chart(inputs, input_values, f'Inputs usage in b={b} exploration pareto front', f'exploration_inputs_usage_B{b}')
    __logger.info("Pareto inputs usage plot for b=%d written successfully", b)


def plot_children_inputs_and_operators_usage(b: int, operators: 'list[str]', inputs: 'list[int]', children_cnn: 'list[str]'):
    op_counters = __initialize_dict_usage_data(operators)
    input_counters = __initialize_dict_usage_data(inputs)

    __logger.info("Analyzing operators and inputs usage of CNN children to train for b=%d", b)
    op_counters, input_counters = __update_counters(children_cnn, op_counters, input_counters)
    op_values = __generate_value_list_from_op_counters_dict(op_counters, operators)
    input_values = __generate_value_list_from_inputs_counters_dict(input_counters, inputs)

    __plot_pie_chart(operators, op_values, f'Operations usage in b={b} CNN children', f'children_op_usage_B{b}')
    __logger.info("Children operators usage plot for b=%d written successfully", b)
    __plot_pie_chart(inputs, input_values, f'Inputs usage in b={b} CNN children', f'children_inputs_usage_B{b}')
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


def plot_predictions_error(B: int, K: int, pnas_mode: bool):
    time_errors, avg_time_errors, max_time_errors, min_time_errors = [], np.zeros(B - 1), np.zeros(B - 1), np.zeros(B - 1)
    acc_errors, avg_acc_errors, max_acc_errors, min_acc_errors = [], np.zeros(B - 1), np.zeros(B - 1), np.zeros(B - 1)
    time_mapes, acc_mapes, time_spearman_coeffs, acc_spearman_coeffs = np.zeros(B - 1), np.zeros(B - 1), np.zeros(B - 1), np.zeros(B - 1)
    time_r2, acc_r2 = np.zeros(B - 1), np.zeros(B - 1)

    # They are used as list of lists for scatter plots.
    pred_times, real_times = [], []
    pred_acc, real_acc = [], []
    scatter_time_legend_labels, scatter_acc_legend_labels = [], []

    for b in range(2, B + 1):
        __logger.info("Comparing predicted values with actual CNN training of b=%d", b)
        merge_df = __build_prediction_dataframe(b, K, pnas_mode)

        # compute time prediction errors (regressor)
        if not pnas_mode:
            pred_times.append(merge_df['time'].to_list())
            real_times.append(merge_df['training time(seconds)'].to_list())

            time_errors_b = merge_df['training time(seconds)'] - merge_df['time']
            time_errors.append(time_errors_b)

            avg_time_errors[b - 2] = statistics.mean(time_errors_b)
            max_time_errors[b - 2] = max(time_errors_b)
            min_time_errors[b - 2] = min(time_errors_b)

            time_mapes[b - 2] = compute_mape(merge_df['training time(seconds)'].to_list(), merge_df['time'].to_list())
            time_spearman_coeffs[b - 2] = compute_spearman_rank_correlation_coefficient_from_df(merge_df, 'training time(seconds)', 'time')
            time_r2[b - 2] = r2_score(merge_df['training time(seconds)'].to_list(), merge_df['time'].to_list())

        # always compute accuracy prediction errors (LSTM controller)
        pred_acc.append(merge_df['val accuracy'].to_list())
        real_acc.append(merge_df['best val accuracy'].to_list())

        val_accuracy_errors_b = merge_df['best val accuracy'] - merge_df['val accuracy']
        acc_errors.append(val_accuracy_errors_b)

        avg_acc_errors[b - 2] = statistics.mean(val_accuracy_errors_b)
        max_acc_errors[b - 2] = max(val_accuracy_errors_b)
        min_acc_errors[b - 2] = min(val_accuracy_errors_b)

        acc_mapes[b - 2] = compute_mape(merge_df['best val accuracy'].to_list(), merge_df['val accuracy'].to_list())
        acc_spearman_coeffs[b - 2] = compute_spearman_rank_correlation_coefficient_from_df(merge_df, 'best val accuracy', 'val accuracy')
        acc_r2[b - 2] = r2_score(merge_df['best val accuracy'].to_list(), merge_df['val accuracy'].to_list())

        # add MAPE, R^2 and spearman metrics to legends
        scatter_time_legend_labels.append(f'B{b} (MAPE: {time_mapes[b - 2]:.3f}%, ρ: {time_spearman_coeffs[b - 2]:.3f})')
        scatter_acc_legend_labels.append(f'B{b} (MAPE: {acc_mapes[b - 2]:.3f}%, ρ: {acc_spearman_coeffs[b - 2]:.3f})')

    x = np.arange(2, B + 1)

    # write plots about time
    if not pnas_mode:
        time_bars = __generate_avg_max_min_bars(avg_time_errors, max_time_errors, min_time_errors)

        __plot_multibar_histogram(x, time_bars, 0.15, 'Blocks', 'Time(s)',
                                  'Time prediction errors overview (real - predicted)', 'pred_time_errors_overview')
        __plot_boxplot(time_errors, x, 'Blocks', 'Time error', 'Time prediction errors overview (real - predicted)', 'pred_time_errors_boxplot')
        __plot_squared_scatter_chart(real_times, pred_times, 'Real time(seconds)', 'Predicted time(seconds)', 'Time predictions overview',
                                     'time_pred_overview', legend_labels=scatter_time_legend_labels)

    acc_bars = __generate_avg_max_min_bars(avg_acc_errors, max_acc_errors, min_acc_errors)

    # write plots about accuracy
    __plot_multibar_histogram(x, acc_bars, 0.15, 'Blocks', 'Accuracy',
                              'Val accuracy prediction errors overview (real - predicted)', 'pred_acc_errors_overview')
    __plot_boxplot(acc_errors, x, 'Blocks', 'Accuracy error',
                   'Accuracy prediction errors overview (real - predicted)', 'pred_acc_errors_boxplot')
    __plot_squared_scatter_chart(real_acc, pred_acc, 'Real accuracy', 'Predicted accuracy', 'Accuracy predictions overview',
                                 'acc_pred_overview', legend_labels=scatter_acc_legend_labels)

    __logger.info("Prediction error overview plots written successfully")


def plot_pareto_front_curves(B: int, plot3d: bool = False):
    training_csv_path = log_service.build_path('csv', 'training_results.csv')
    training_df = pd.read_csv(training_csv_path)
    training_df = training_df[training_df['exploration'] == False]

    # front built with actual values got from training
    real_front_acc, real_front_time = [], []
    # pareto front predicted
    pred_front_acc, pred_front_time = [], []

    for b in range(2, B + 1):
        training_df_b = training_df[training_df['# blocks'] == b]
        real_front_acc.append(training_df_b['best val accuracy'].to_list())
        real_front_time.append(training_df_b['training time(seconds)'].to_list())

        pareto_b_csv_path = log_service.build_path('csv', f'pareto_front_B{b}.csv')
        pareto_df = pd.read_csv(pareto_b_csv_path)
        pred_front_acc.append(pareto_df['val accuracy'].to_list())
        pred_front_time.append(pareto_df['time'].to_list())

    b = 2
    for real_time, real_acc, pred_time, pred_acc in zip(real_front_time, real_front_acc, pred_front_time, pred_front_acc):
        plot_func = __plot_3d_pareto_front if plot3d else __plot_pareto_front
        plot_func(real_time, real_acc, pred_time, pred_acc, title=f'Pareto front B{b}', save_name=f'pareto_plot_B{b}')
        b += 1

    __logger.info("Pareto front curve plots written successfully")


def plot_multi_output_boxplot():
    __logger.info("Analyzing accuracy of the multiple outputs for each model...")
    csv_path = log_service.build_path('csv', 'multi_output.csv')
    outputs_df = pd.read_csv(csv_path).drop(columns=['cell_spec'])

    # remove all rows that have some missing values (this means the cell don't use -1 lookback, stacking less cells)
    # removing these rows allow a fair comparison in the plot
    full_df = outputs_df.dropna()
    x_labels = full_df.columns.to_list()
    output_accuracies = [full_df[output_key] for output_key in x_labels]

    # get the entries with less cells
    na_df = outputs_df[outputs_df[x_labels[-2]].isnull()]
    x_labels_na = x_labels[::-2][::-1]
    output_accuracies_na = [na_df[output_key] for output_key in x_labels_na]

    x_labels = [str(label).split('_')[0] for label in x_labels]
    x_labels_na = x_labels[::-2][::-1]

    __plot_boxplot(output_accuracies, x_labels, 'Outputs', 'Val accuracy', 'Best accuracies for output', 'multi_output_boxplot')
    __plot_boxplot(output_accuracies_na, x_labels_na, 'Outputs', 'Val accuracy',
                   'Best accuracies for output (-2 lookback)', 'multi_output_boxplot_na')

    __logger.info("Multi output overview plot written successfully")


class BarInfo(NamedTuple):
    y: TextFileReader  # pandas df column type
    color: str
    label: str
