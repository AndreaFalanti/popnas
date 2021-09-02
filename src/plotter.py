import matplotlib.pyplot as plt
import pandas as pd

import log_service

# Provides utility functions for plotting relevant data gained during the algorithm run,
# so that it can be further analyzed in a more straightforward way

__logger = None

# TODO: otherwise would be initialized before run.py code, producing an error. Is there a less 'hacky' way?
def initialize_logger():
    global __logger
    __logger = log_service.get_logger(__name__)

def __plot_histogram(x, y, x_label, y_label, title, save_name):
    fig = plt.figure()
    plt.bar(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    # beautify the x-labels
    plt.gcf().autofmt_xdate()

    save_path = log_service.build_path('plots', save_name)
    plt.savefig(save_path, bbox_inches='tight')


def plot_dynamic_reindex_related_blocks_info():
    __logger.info("Analyzing training_results.csv...")
    csv_path = log_service.build_path('csv', 'training_results.csv')
    df = pd.read_csv(csv_path)

    # take only mono block cells
    df = df[df['# blocks'] == 1]

    # parse cell structure (trim square brackets and split by ;)
    cells = list(map(lambda cs: cs[1:-1].split(';'), df['cell structure']))

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
    __plot_histogram(x, y_time, 'Operation', 'Training Time (s)', 'SMB -1 input training time', 'SMB_time.png')
    __plot_histogram(x, y_acc, 'Operation', 'Val Accuracy', 'SMB -1 input validation accuracy', 'SMB_acc.png')
    __plot_histogram(x, y_params, 'Operation', 'Params', 'SMB -1 input total parameters', 'SMB_params.png')
    __logger.info("SMB plots written successfully")
