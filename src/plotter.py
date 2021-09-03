from typing import Iterable, NamedTuple

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from pandas.io.parsers import TextFileReader

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

    # beautify the x-labels
    plt.gcf().autofmt_xdate()

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
    
    ax.legend()
    
    save_path = log_service.build_path('plots', save_name)
    plt.savefig(save_path, bbox_inches='tight')

def __plot_pie_chart(labels, values, save_name):
    total = sum(values)
    def pct_val_formatter(x):
        return '{:.3f}%\n({:.0f})'.format(x, total*x/100)

    fig1, ax1 = plt.subplots()

    explode = np.empty(len(labels)) # type: np.ndarray
    explode.fill(0.1)

    ax1.pie(values, labels=labels, autopct=pct_val_formatter, explode=explode, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

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


def __prune_zero_values_and_labels(operations, op_counters):
    '''
    Prune labels associated to 0 values to avoid displaying them in plot
    '''
    for index, val in enumerate(op_counters):
        if val == 0:
            operations.pop(index)

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

    operations, op_counters = __prune_zero_values_and_labels(operations, op_counters)

    __plot_pie_chart(operations, op_counters, f'pareto_op_usage_B{b}')
    __logger.info("Pareto op usage plot for b=%d written successfully", b)


def plot_children_op_usage(b: int, operations: 'list[str]', children_cnn: 'list[str]'):
    op_index, op_counters = __initialize_operation_usage_data(operations)

    __logger.info("Analyzing operation usage for CNN children to train for b=%d", b)
    op_counters = __update_op_counters(children_cnn, b, op_counters, op_index)

    operations, op_counters = __prune_zero_values_and_labels(operations, op_counters)

    __plot_pie_chart(operations, op_counters, f'children_op_usage_B{b}')
    __logger.info("Children op usage plot for b=%d written successfully", b)


class BarInfo(NamedTuple):
    y: TextFileReader   # pandas df column type
    color: str
    label: str
