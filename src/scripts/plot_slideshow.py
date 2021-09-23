import argparse
import math
import os
from sys import platform

import matplotlib.pyplot as plt
from PIL import Image


def path_closure(log_folder):
    plot_path = os.path.join(log_folder, 'plots')

    def gen_paths(paths):
        return list(map(lambda path: os.path.join(plot_path, path), paths))

    return gen_paths


def display_plot_overview(plot_paths, columns, rows, title=None):
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

    # maximize graph (TKAgg backend)
    mng = plt.get_current_fig_manager()
    if platform == 'win32':
        mng.window.state('zoomed')
    else:
        mng.resize(*mng.window.maxsize())

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-p', metavar='FOLDER', type=str, help="log folder", required=True)
    args = parser.parse_args()

    gen_paths = path_closure(args.p)

    display_plot_overview(gen_paths(['SMB_acc.png', 'SMB_time.png', 'SMB_params.png', 'SMB_flops.png']), 2, 2,
                          title='Specular mono blocks (input -1) overview')
    display_plot_overview(
        gen_paths(['acc_pred_overview.png', 'pred_acc_errors_overview.png', 'time_pred_overview.png', 'pred_time_errors_overview.png']),
        2, 2, title='Prediction errors overview')

    b = 2
    while os.path.isfile(os.path.join(args.p, 'plots', f'children_op_usage_B{b}.png')):
        display_plot_overview(gen_paths([f'pareto_op_usage_B{b}.png', f'children_op_usage_B{b}.png']), 2, 1,
                              title=f'Operation usage overview (B={b})')
        b += 1

    display_plot_overview(gen_paths(['train_time_overview.png', 'train_acc_overview.png']), 2, 1, title='CNN training per block overview')

    # check if regressor test folder is present (regressor_testing.py output)
    reg_test_path = os.path.join(args.p, 'regressors_test')
    if os.path.isdir(reg_test_path):
        subfolders_full_path = [f.path for f in os.scandir(reg_test_path) if f.is_dir()]
        regressors_num = len(subfolders_full_path)

        regressor_plot_paths = []
        for subfolder in subfolders_full_path:
            plot_path = os.path.join(subfolder, 'results.png')
            regressor_plot_paths.append(plot_path)

        display_plot_overview(regressor_plot_paths, math.ceil(regressors_num / 2.0), regressors_num // 2, title='Regressor testing overview')


if __name__ == '__main__':
    main()
