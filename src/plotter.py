from typing import Iterable, NamedTuple

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from pandas.io.parsers import TextFileReader
import statistics

import log_service

# Provides utility functions for plotting relevant data gained during the algorithm run,
# so that it can be further analyzed in a more straightforward way

__logger = None

# TODO: otherwise would be initialized before run.py code, producing an error. Is there a less 'hacky' way?
def initialize_logger():
    global __logger
    __logger = log_service.get_logger(__name__)


def __parse_cell_structures(cell_structures: Iterable):
    # parse cell structure (trim square brackets and split by ;)
    return  list(map(lambda cs: cs[1:-1].split(';'), cell_structures))


def __plot_histogram(x, y, x_label, y_label, title, save_name):
    plt.figure()
    plt.bar(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    # add y grid lines
    plt.grid(b=True, which='both', axis='y', alpha=0.5, color='k')

    # beautify the x-labels
    # plt.gcf().autofmt_xdate()

    save_path = log_service.build_path('plots', save_name)
    plt.savefig(save_path, bbox_inches='tight')


def __plot_multibar_histogram(x, y_array: 'list[BarInfo]', col_width, x_label, y_label, title, save_name):
    fig, ax = plt.subplots()
    
    x_label_dist = np.arange(len(x))  # the label locations
    x_offset = - ((len(y_array) - 1) / 2.0) * col_width

    # Make the plot
    for bar_info in y_array:
        ax.bar(x_label_dist + x_offset, bar_info.y, color=bar_info.color, width = col_width, label=bar_info.label)
        x_offset += col_width
    
    ax.set_xticks(x_label_dist)
    ax.set_xticklabels(x)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    
    # add y grid lines
    plt.grid(b=True, which='both', axis='y', alpha=0.5, color='silver')

    ax.legend()
    
    save_path = log_service.build_path('plots', save_name)
    plt.savefig(save_path, bbox_inches='tight')


def __plot_pie_chart(labels, values, title, save_name):
    total = sum(values)
    # def pct_val_formatter(x):
    #     return '{:.3f}%\n({:.0f})'.format(x, total*x/100)

    fig1, ax1 = plt.subplots()

    explode = np.empty(len(labels)) # type: np.ndarray
    explode.fill(0.03)

    # label, percentage, value are written only in legend, to avoid overlapping texts in chart
    legend_labels = [f'{label} - {(val/total)*100:.3f}% ({val:.0f})' for label, val in zip(labels, values)]

    patches, texts = ax1.pie(values, labels=labels, explode=explode, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.title(title)
    plt.legend(patches, legend_labels, loc='lower left', bbox_to_anchor=(1.05, 0.1))
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.75)

    save_path = log_service.build_path('plots', save_name)
    plt.savefig(save_path, bbox_inches='tight')


def __plot_squared_scatter_chart(x, y, x_label, y_label,  title, save_name, plot_reference = True):
    fig, ax = plt.subplots()
    plt.scatter(x, y)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.gca().set_aspect('equal', adjustable='box')

    if plot_reference:
        ax_lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]

        ax.plot(ax_lims, ax_lims, '--k', alpha=0.75)

    save_path = log_service.build_path('plots', save_name)
    plt.savefig(save_path, bbox_inches='tight')


def plot_dynamic_reindex_related_blocks_info():
    __logger.info("Analyzing training_results.csv...")
    csv_path = log_service.build_path('csv', 'training_results.csv')
    df = pd.read_csv(csv_path)

    # take only mono block cells
    df = df[df['# blocks'] == 1]

    cells = __parse_cell_structures(df['cell structure'])

    # TODO: refactor this mess if you find a more intelligent way
    df['in1'] = [x for x, _, _, _ in cells]
    df['op1'] = [x for _, x, _, _ in cells]
    df['in2'] = [x for _, _, x, _ in cells]
    df['op2'] = [x for _, _, _, x in cells]

    df = df[(df['in1'] == df['in2']) & (df['op1'] == df['op2']) & (df['in1'] == '-1')]

    x = df['op1']
    y_time = df['training time(seconds)']
    y_acc = df['best val accuracy']
    y_params = df['total params']
  
    __logger.info("Writing plots...")
    __plot_histogram(x, y_time, 'Operation', 'Time(s)', 'SMB -1 input training time', 'SMB_time.png')
    __plot_histogram(x, y_acc, 'Operation', 'Val Accuracy', 'SMB -1 input validation accuracy', 'SMB_acc.png')
    __plot_histogram(x, y_params, 'Operation', 'Params', 'SMB -1 input total parameters', 'SMB_params.png')
    __logger.info("SMB plots written successfully")


def plot_training_info_per_block():
    __logger.info("Analyzing avg_training_time.csv...")
    csv_path = log_service.build_path('csv', 'avg_training_time.csv')
    df = pd.read_csv(csv_path)

    x = df['# blocks']
    bar_avg = BarInfo(df['avg training time(s)'], 'b', 'avg') 
    bar_max = BarInfo(df['max time'], 'g', 'max')
    bar_min = BarInfo(df['min time'], 'r', 'min')

    __plot_multibar_histogram(x, [bar_avg, bar_max, bar_min], 0.15, 'Blocks', 'Time(s)', 'Training time overview', 'train_time_overview.png')
    __logger.info("Train time overview plot written successfully")


def __initialize_operation_usage_data(operations):
    '''
    Create dictionary with indexes and initialize counters.
    '''
    op_index = {}
    op_counters = np.zeros(len(operations)) # type: np.ndarray

    for index, op in enumerate(operations):
        op_index[op] = index

    return op_index, op_counters


def __prune_zero_values_and_labels(operations, op_counters: np.ndarray):
    '''
    Prune labels associated to 0 values to avoid displaying them in plot
    '''
    op_gather_indexes = np.flatnonzero(op_counters)
    operations = [operations[i] for i in op_gather_indexes]

    # prune 0 values
    op_counters = op_counters[op_counters != 0]

    return operations, op_counters


def __update_op_counters(cells, b, op_counters, op_index):
    '''
    Iterate cell structures and increment operation counters when the operation is encountered. 
    '''
    # iterate all cells (models selected for training)
    for cell in cells:
        # iterate on blocks (in1, op1, in2, op2)
        for i in range(b):
            # get indices for operation in dict 
            op1_index = op_index[cell[(i*4) + 1]]
            op2_index = op_index[cell[(i*4) + 3]]

            op_counters[op1_index] += 1
            op_counters[op2_index] += 1

    return op_counters


def plot_operation_usage(b: int, operations: 'list[str]'):
    op_index, op_counters = __initialize_operation_usage_data(operations)

    __logger.info("Analyzing operation usage for pareto front of b=%d", b)
    csv_path = log_service.build_path('csv', f'pareto_front_B{b}.csv')
    df = pd.read_csv(csv_path)

    cells = __parse_cell_structures(df['cell structure'])

    op_counters = __update_op_counters(cells, b, op_counters, op_index)

    #operations, op_counters = __prune_zero_values_and_labels(operations, op_counters)

    __plot_pie_chart(operations, op_counters, f'Operations usage in b={b} pareto front', f'pareto_op_usage_B{b}')
    __logger.info("Pareto op usage plot for b=%d written successfully", b)


def plot_children_op_usage(b: int, operations: 'list[str]', children_cnn: 'list[str]'):
    op_index, op_counters = __initialize_operation_usage_data(operations)

    __logger.info("Analyzing operation usage for CNN children to train for b=%d", b)
    op_counters = __update_op_counters(children_cnn, b, op_counters, op_index)

    #operations, op_counters = __prune_zero_values_and_labels(operations, op_counters)

    __plot_pie_chart(operations, op_counters, f'Operations usage in b={b} CNN children', f'children_op_usage_B{b}')
    __logger.info("Children op usage plot for b=%d written successfully", b)


def plot_predictions_error(B: int):
    avg_time_errors, max_time_errors, min_time_errors = np.zeros(B-1), np.zeros(B-1), np.zeros(B-1)
    avg_acc_errors, max_acc_errors, min_acc_errors = np.zeros(B-1), np.zeros(B-1), np.zeros(B-1)

    # TODO. maybe better to refactor to numpy arrays too
    pred_times, real_times = [], []
    pred_acc, real_acc = [], []

    for b in range(2, B+1):
        __logger.info("Comparing predicted values with actual CNN training of b=%d", b)
        pareto_csv_path = log_service.build_path('csv', f'pareto_front_B{b}.csv')
        training_csv_path = log_service.build_path('csv', 'training_results.csv')
        pareto_df = pd.read_csv(pareto_csv_path)
        training_df = pd.read_csv(training_csv_path)

        # take only trained CNN with correct block length
        training_df = training_df[training_df['# blocks'] == b]

        # take first k CNN from pareto front (k can be found as the actual trained CNN count)
        trained_cnn_count = len(training_df)
        pareto_df = pareto_df.head(trained_cnn_count)

        # now both dataframes have same length and they have the same CNN order. Confront items in order to get differences.
        merge_df = pd.merge(training_df, pareto_df, on=['cell structure'], how='inner')
        time_errors = merge_df['training time(seconds)'] - merge_df['time']
        val_accuracy_errors = merge_df['best val accuracy'] - merge_df['val accuracy']

        pred_times += merge_df['time'].to_list()
        real_times += merge_df['training time(seconds)'].to_list()
        pred_acc += merge_df['val accuracy'].to_list()
        real_acc += merge_df['best val accuracy'].to_list()

        avg_time_errors[b-2] = statistics.mean(time_errors)
        max_time_errors[b-2] = max(time_errors)
        min_time_errors[b-2] = min(time_errors)

        avg_acc_errors[b-2] = statistics.mean(val_accuracy_errors)
        max_acc_errors[b-2] = max(val_accuracy_errors)
        min_acc_errors[b-2] = min(val_accuracy_errors)

    x = np.arange(2, B+1)

    bar_avg_t = BarInfo(avg_time_errors, 'b', 'avg') 
    bar_max_t = BarInfo(max_time_errors, 'g', 'max')
    bar_min_t = BarInfo(min_time_errors, 'r', 'min')

    bar_avg_acc = BarInfo(avg_acc_errors, 'b', 'avg') 
    bar_max_acc = BarInfo(max_acc_errors, 'g', 'max')
    bar_min_acc = BarInfo(min_acc_errors, 'r', 'min')

    __plot_multibar_histogram(x, [bar_avg_t, bar_max_t, bar_min_t], 0.15, 'Blocks', 'Time(s)',
                                'Predictions time errors overview (real - predicted)', 'pred_time_errors_overview.png')
    __plot_multibar_histogram(x, [bar_avg_acc, bar_max_acc, bar_min_acc], 0.15, 'Blocks', 'Accuracy',
                                'Predictions val accuracy errors overview (real - predicted)', 'pred_acc_errors_overview.png')
    __plot_squared_scatter_chart(real_times, pred_times, 'Real time(seconds)', 'Predicted time(seconds)', 'Time predictions overview', 'time_pred_overview.png')
    __plot_squared_scatter_chart(real_acc, pred_acc, 'Real accuracy', 'Predicted accuracy', 'Accuracy predictions overview', 'acc_pred_overview.png')

    __logger.info("Prediction error overview plots written successfully")


class BarInfo(NamedTuple):
    y: TextFileReader   # pandas df column type
    color: str
    label: str
