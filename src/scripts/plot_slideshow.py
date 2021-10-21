import argparse
import math
import os
import shutil
from sys import platform

import matplotlib.pyplot as plt
from PIL import Image

from utils.func_utils import clamp


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


def display_plot_overview(plot_paths, columns, rows, title=None, save=False, save_name=None):
    assert len(plot_paths) <= (columns * rows)

    # force TkAgg for maximizing the window with code below
    plt.switch_backend('TkAgg')
    fig, ax = plt.subplots(nrows=rows, ncols=columns)

    for i, axi in enumerate(ax.flat):
        # disable axis visualization around image
        axi.axis('off')

        if i < len(plot_paths):
            try:
                with Image.open(plot_paths[i]) as img:
                    axi.imshow(img)
            except FileNotFoundError:
                axi.text(0.5, 0.5, s='Image not found', ha='center', va='center', fontsize='x-large', fontweight='semibold')

    plt.tight_layout()

    if title is not None:
        fig.suptitle(title, fontsize='x-large', fontweight='semibold', y=1.0)

    # if in save mode, just save plot to file
    if save:
        fig.savefig(save_name, bbox_inches='tight', dpi=300)
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
    cols = clamp(math.ceil(plots_num / 2.0), 0, 4)
    rows = math.ceil(plots_num / cols)

    return cols, rows


def display_predictors_test_slide(test_folder_path: str, reference_plot_path: str, title: str, save: bool = False, save_name: str = None):
    subfolders_full_path = [f.path for f in os.scandir(test_folder_path) if f.is_dir()]

    predictors_plot_paths = []
    for subfolder in subfolders_full_path:
        plot_path = os.path.join(subfolder, 'results.png')
        predictors_plot_paths.append(plot_path)

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

    display_plot_overview(gen_paths(['SMB_acc.png', 'SMB_time.png', 'SMB_params.png', 'SMB_flops.png']),
                          2, 2, title='Specular mono blocks (input -1) overview', save=args.save, save_name=next(gen_save_path, None))
    display_plot_overview(gen_paths(['acc_pred_overview.png', 'pred_acc_errors_boxplot.png',
                                     'time_pred_overview.png', 'pred_time_errors_boxplot.png']),
                          2, 2, title='Prediction errors overview', save=args.save, save_name=next(gen_save_path, None))

    b = 2
    while os.path.isfile(os.path.join(args.p, 'plots', f'children_op_usage_B{b}.png')):
        display_plot_overview(gen_paths([f'pareto_op_usage_B{b}.png', f'children_op_usage_B{b}.png',
                                         f'pareto_inputs_usage_B{b}.png', f'children_inputs_usage_B{b}.png']),
                              2, 2, title=f'Cell structures overview (B={b})', save=args.save, save_name=next(gen_save_path, None))
        b += 1

    pareto_plot_paths = [filename for filename in os.listdir(os.path.join(args.p, 'plots')) if filename.startswith('pareto_plot_B')]
    if len(pareto_plot_paths) > 0:
        cols, rows = compute_dynamic_size_layout(len(pareto_plot_paths))
        display_plot_overview(gen_paths(pareto_plot_paths), cols, rows, title='Pareto fronts overview', save=args.save, save_name=next(gen_save_path, None))

    display_plot_overview(gen_paths(['train_time_overview.png', 'train_acc_overview.png', 'train_time_boxplot.png', 'val_acc_boxplot.png']),
                          2, 2, title='CNN training per block overview', save=args.save, save_name=next(gen_save_path, None))

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
