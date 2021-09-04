import argparse
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
    fig, ax = plt.subplots(nrows=rows, ncols=columns)

    for i, axi in enumerate(ax.flat):
        # disable axis visualization around image
        axi.axis('off')

        if i < len(plot_paths):
            with Image.open(plot_paths[i]) as img:
                axi.axis('off')
                axi.imshow(img)

    plt.tight_layout()

    if title is not None:
        fig.suptitle(title, fontsize='x-large', fontweight='semibold')

    # maximize graph
    mng  = plt.get_current_fig_manager()
    if platform == 'win32':
        mng.window.state('zoomed')
    else:
        mng.resize(*mng.window.maxsize())

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-p', metavar=('FOLDER'), type=str, help="log folder", required=True)
    args = parser.parse_args()

    gen_paths = path_closure(args.p)

    display_plot_overview(gen_paths(['SMB_acc.png', 'SMB_time.png', 'SMB_params.png']), 3, 1, title='Specular mono blocks (input -1) overview')
    display_plot_overview(gen_paths(['acc_pred_overview.png', 'pred_acc_errors_overview.png', 'time_pred_overview.png', 'pred_time_errors_overview.png']), 2, 2,
                            title='Prediction errors overview')

    b = 2
    while os.path.isfile(os.path.join(args.p, 'plots', f'pareto_op_usage_B{b}.png')):
        display_plot_overview(gen_paths([f'pareto_op_usage_B{b}.png', f'children_op_usage_B{b}.png']), 2, 1, title=f'Operation usage overview (B={b})')
        b += 1

    display_plot_overview(gen_paths(['train_time_overview.png']), 2, 1)

if __name__ == '__main__':
    main()
