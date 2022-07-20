import argparse
import json
import math
import os
import shutil
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

    def gen_save_name():
        for i in range(1, 1000):
            yield os.path.join(save_folder, f'slide_{i}.png')

    return gen_save_name()


def display_plot_overview(plot_paths: 'list[str]', columns: int, rows: int, title: str = None, save: bool = False, save_name: str = None):
    assert len(plot_paths) <= (columns * rows)

    # force TkAgg for maximizing the window with code below
    plt.switch_backend('TkAgg')
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
        fig.savefig(save_name, bbox_inches='tight', dpi=120)
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


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-p', metavar='FOLDER', type=str, help="log folder", required=True)
    parser.add_argument('--save', help="save slides into a folder, instead of displaying them", action="store_true")
    args = parser.parse_args()

    gen_paths = path_closure(args.p)
    gen_save_path = generate_slide_save_path(args.p) if args.save else iter(())

    with open(os.path.join(args.p, 'restore', 'run.json')) as f:
        run_config = json.load(f)

    sstr_config = run_config['search_strategy']
    score_metric = 'accuracy' if 'accuracy' in sstr_config['pareto_objectives'] else 'f1_score'

    display_plot_overview(gen_paths(['SMB_acc.png', 'SMB_time.png', 'SMB_params.png', 'SMB_flops.png']),
                          2, 2, title='Specular mono blocks (input -1) overview', save=args.save, save_name=next(gen_save_path, None))
    display_plot_overview(gen_paths([f'{score_metric}_pred_overview.png', f'pred_{score_metric}_errors_boxplot.png',
                                     'time_pred_overview.png', 'pred_time_errors_boxplot.png']),
                          2, 2, title='Prediction errors overview', save=args.save, save_name=next(gen_save_path, None))

    b = 2
    while os.path.isfile(os.path.join(args.p, 'plots', f'children_op_usage_B{b}.png')):
        display_plot_overview(gen_paths([f'pareto_op_usage_B{b}.png', f'exploration_op_usage_B{b}.png', f'children_op_usage_B{b}.png',
                                         f'pareto_inputs_usage_B{b}.png', f'exploration_inputs_usage_B{b}.png', f'children_inputs_usage_B{b}.png']),
                              3, 2, title=f'Cell structures overview (B={b})', save=args.save, save_name=next(gen_save_path, None))
        b += 1

    pred_pareto_plot_paths = [filename for filename in os.listdir(os.path.join(args.p, 'plots')) if filename.startswith('predictions_with_pareto_B')]
    if len(pred_pareto_plot_paths) > 0:
        cols, rows = compute_dynamic_size_layout(len(pred_pareto_plot_paths))
        display_plot_overview(gen_paths(pred_pareto_plot_paths), cols, rows, title='Predictions with Pareto overview', save=args.save,
                              save_name=next(gen_save_path, None))

    real_pareto_plot_paths = [filename for filename in os.listdir(os.path.join(args.p, 'plots')) if filename.startswith('pareto_plot_B')]
    if len(real_pareto_plot_paths) > 0:
        cols, rows = compute_dynamic_size_layout(len(real_pareto_plot_paths))
        display_plot_overview(gen_paths(real_pareto_plot_paths), cols, rows, title='Pareto fronts overview', save=args.save,
                              save_name=next(gen_save_path, None))

    multi_output_plot_paths = [filename for filename in os.listdir(os.path.join(args.p, 'plots')) if filename.startswith('multi_output_boxplot')]
    if len(multi_output_plot_paths) > 0:
        cols, rows = compute_dynamic_size_layout(len(multi_output_plot_paths))
        display_plot_overview(gen_paths(multi_output_plot_paths), cols, rows, title='Val accuracy overview per output', save=args.save,
                              save_name=next(gen_save_path, None))

    display_plot_overview(gen_paths(['train_time_overview.png', 'train_acc_overview.png', 'train_time_boxplot.png', 'val_acc_boxplot.png']),
                          2, 2, title='CNN training per block overview', save=args.save, save_name=next(gen_save_path, None))

    time_corr_plot_paths = [filename for filename in os.listdir(os.path.join(args.p, 'plots')) if filename.endswith('_time_corr.png')]
    if len(time_corr_plot_paths) > 0:
        cols, rows = compute_dynamic_size_layout(len(time_corr_plot_paths))
        display_plot_overview(gen_paths(time_corr_plot_paths), cols, rows, title='Correlation with training time overview', save=args.save,
                              save_name=next(gen_save_path, None))

    # check if predictors time test folder is present (predictors_time_testing.py output)
    time_test_path = os.path.join(args.p, 'pred_time_test')
    if os.path.isdir(time_test_path):
        display_predictors_test_slide(time_test_path, *gen_paths(['time_pred_overview.png']), title='Time prediction testing overview',
                                      save=args.save, save_name=next(gen_save_path, None))

    # check if predictors acc test folder is present (predictors_acc_testing.py output)
    acc_test_path = os.path.join(args.p, 'pred_acc_test')
    if os.path.isdir(acc_test_path):
        display_predictors_test_slide(acc_test_path, *gen_paths(['acc_pred_overview.png']), title='Accuracy prediction testing overview',
                                      save=args.save, save_name=next(gen_save_path, None))


if __name__ == '__main__':
    main()
