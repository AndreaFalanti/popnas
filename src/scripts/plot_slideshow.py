import argparse
import math
import os
import shutil
from operator import attrgetter
from sys import platform

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from utils.func_utils import clamp, chunks


def path_closure(log_folder):
    plot_path = os.path.join(log_folder, 'plots')

    def gen_paths(paths):
        return list(map(lambda path: os.path.join(plot_path, path), paths))

    return gen_paths


def generate_slide_save_path(log_folder):
    save_folder = os.path.join(log_folder, 'plot_slides')
    try:
        os.makedirs(save_folder)
    except OSError:
        shutil.rmtree(save_folder)
        os.makedirs(save_folder)

    for i in range(1, 1000):
        yield os.path.join(save_folder, f'slide_{i}.png')


def display_plot_overview(plot_paths: 'list[str]', columns: int, rows: int, title: str = None, save: bool = False, save_name: str = None):
    assert len(plot_paths) <= (columns * rows)

    # basically use an aspect ratio of 16:9, figsize is in inches and pixels for inches are given by dpi
    fig, ax = plt.subplots(nrows=rows, ncols=columns, figsize=(16, 9), dpi=100)  # type: plt.Figure, np.ndarray[plt.Axes]
    fig.tight_layout(pad=2.0)

    for i, axi in enumerate(ax.flat):
        # disable axis visualization around image
        axi = axi  # type: plt.Axes
        axi.axis('off')

        if i < len(plot_paths):
            try:
                with Image.open(plot_paths[i]) as img:
                    axi.imshow(img)
            except FileNotFoundError:
                axi.text(0.5, 0.5, s='Image not found', ha='center', va='center', fontsize='medium', fontweight='semibold')

    plt.tight_layout()

    if title is not None:
        fig.suptitle(title, fontsize='x-large', fontweight='semibold', y=1.0)

    # if in save mode, just save plot to file
    if save:
        fig.savefig(save_name, bbox_inches='tight', dpi=150)
    # if not in save mode, display the plot on screen
    else:
        # maximize graph (TKAgg backend)
        mng = plt.get_current_fig_manager()
        if platform == 'win32':
            mng.window.state('zoomed')
        else:
            mng.resize(*mng.window.maxsize())

        plt.show()


def compute_dynamic_size_layout(plots_num: int):
    rows = clamp(math.ceil(plots_num / 2.0), 1, 2)
    cols = math.ceil(plots_num / rows)

    return cols, rows


def display_predictors_test_slide(test_folder_path: str, reference_plot_path: str, title: str, save: bool = False, save_name: str = None):
    subfolders_full_path = [f.path for f in os.scandir(test_folder_path) if f.is_dir()]

    predictors_plot_paths = []
    for subfolder in subfolders_full_path:
        plot_path = os.path.join(subfolder, 'results.png')
        predictors_plot_paths.append(plot_path)

    # split into multiple slides if there are too many plots for good visualization
    chunk_size = 5
    if len(predictors_plot_paths) > chunk_size:
        for i, plot_subgroup in enumerate(chunks(predictors_plot_paths, chunk_size)):
            # add reference plot (actual run results) and compute layout info
            plot_subgroup.insert(0, reference_plot_path)
            cols, rows = compute_dynamic_size_layout(len(plot_subgroup))
            subgroup_title = f'{title} ({i})'
            subgroup_save_name = None if save_name is None else f'{save_name.split(".")[0]}_{i}.png'

            display_plot_overview(plot_subgroup, cols, rows, title=subgroup_title, save=save, save_name=subgroup_save_name)
    # plots len fit a single slide
    else:
        # add reference plot (actual run results) and compute layout info
        predictors_plot_paths.insert(0, reference_plot_path)
        cols, rows = compute_dynamic_size_layout(len(predictors_plot_paths))

        display_plot_overview(predictors_plot_paths, cols, rows, title=title, save=save, save_name=save_name)


def display_dynamic_layout_slide(plots: 'list[os.DirEntry]', title: str, save: bool, save_path: str):
    if len(plots) > 1:
        cols, rows = compute_dynamic_size_layout(len(plots))
        plot_paths = [f.path for f in plots]
        display_plot_overview(plot_paths, cols, rows, title, save=save, save_name=save_path)
    else:
        print('Slide skipped, since composed by a single (or zero) plot')


def execute(p: str, save: bool = False):
    ''' Refer to argparse help for more information about these arguments. '''
    if not save:
        # force TkAgg backend for getting an interactive interface and maximizing the window
        plt.switch_backend('TkAgg')

    gen_paths = path_closure(p)
    gen_save_path = generate_slide_save_path(p) if save else iter(())

    # get info about all plots belonging to the apposite folder
    plot_files = [f for f in os.scandir(os.path.join(p, 'plots')) if f.is_file()]  # type: list[os.DirEntry]
    # make sure it is ordered by filename (depends on OS, on Windows it seems already done by scandir, but not on Linux)
    plot_files.sort(key=attrgetter('name'))

    smb_plots = [f for f in plot_files if f.name.startswith('SMB_')]
    display_dynamic_layout_slide(smb_plots, title='Specular mono blocks (input -1) overview', save=save, save_path=next(gen_save_path, None))

    exp_pred_plots = [f for f in plot_files if f.name.endswith('pred_overview.png')]
    exp_pred_errors_plots = [f for f in plot_files if f.name.endswith('errors_boxplot.png')]
    display_dynamic_layout_slide(exp_pred_plots + exp_pred_errors_plots, title='Prediction errors overview',
                                 save=save, save_path=next(gen_save_path, None))

    b = 2
    while os.path.isfile(os.path.join(p, 'plots', f'children_op_usage_B{b}.png')):
        display_plot_overview(gen_paths([f'pareto_op_usage_B{b}.png', f'exploration_op_usage_B{b}.png', f'children_op_usage_B{b}.png',
                                         f'pareto_inputs_usage_B{b}.png', f'exploration_inputs_usage_B{b}.png', f'children_inputs_usage_B{b}.png']),
                              3, 2, title=f'Cell structures overview (B={b})', save=save, save_name=next(gen_save_path, None))
        b += 1

    pred_pareto_plots = [f for f in plot_files if f.name.startswith('predictions_with_pareto_B')]
    display_dynamic_layout_slide(pred_pareto_plots, title='Predictions with Pareto overview', save=save, save_path=next(gen_save_path, None))

    real_pareto_plots = [f for f in plot_files if f.name.startswith('pareto_plot_B')]
    display_dynamic_layout_slide(real_pareto_plots, title='Pareto fronts overview', save=save, save_path=next(gen_save_path, None))

    multi_output_plots = [f for f in plot_files if f.name.startswith('multi_output_')]
    display_dynamic_layout_slide(multi_output_plots, title='Metrics overview per output', save=save, save_path=next(gen_save_path, None))

    metrics_block_boxplots = [f for f in plot_files if f.name.endswith('boxplot.png') and 'errors' not in f.name]
    display_dynamic_layout_slide(metrics_block_boxplots, title='CNN training per block overview', save=save, save_path=next(gen_save_path, None))

    time_corr_plots = [f for f in plot_files if f.name.endswith('_time_corr.png')]
    display_dynamic_layout_slide(time_corr_plots, title='Correlation with training time overview', save=save, save_path=next(gen_save_path, None))

    # check if predictors time test folder is present (predictors_time_testing.py output)
    time_test_path = os.path.join(p, 'pred_time_test')
    if os.path.isdir(time_test_path):
        display_predictors_test_slide(time_test_path, *gen_paths(['time_pred_overview.png']), title='Time prediction testing overview',
                                      save=save, save_name=next(gen_save_path, None))

    # check if predictors acc test folder is present (predictors_acc_testing.py output)
    acc_test_path = os.path.join(p, 'pred_acc_test')
    if os.path.isdir(acc_test_path):
        display_predictors_test_slide(acc_test_path, *gen_paths(['acc_pred_overview.png']), title='Accuracy prediction testing overview',
                                      save=save, save_name=next(gen_save_path, None))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-p', metavar='FOLDER', type=str, help="log folder", required=True)
    parser.add_argument('--save', help="save slides into a folder, instead of displaying them", action="store_true")
    args = parser.parse_args()
    
    execute(**vars(args))
