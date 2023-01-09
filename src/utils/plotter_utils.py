from typing import Optional, Collection, Sequence, NamedTuple

import numpy as np
from matplotlib import pyplot as plt, cm as cm

import log_service


class BarInfo(NamedTuple):
    y: Sequence
    color: str
    label: str


def generate_avg_max_min_bars(avg_vals: Sequence, max_vals: Sequence, min_vals: Sequence) -> 'list[BarInfo, BarInfo, BarInfo]':
    '''
    Build avg, max and min bars for multi-bar plots.

    Args:
        avg_vals: values to assign to avg bar
        max_vals: values to assign to max bar
        min_vals: values to assign to min bar

    Returns:
        BarInfos usable in multi-bar plots
    '''
    bar_avg = BarInfo(avg_vals, 'b', 'avg')
    bar_max = BarInfo(max_vals, 'g', 'max')
    bar_min = BarInfo(min_vals, 'r', 'min')

    return [bar_avg, bar_max, bar_min]


def _save_and_close_plot(fig: plt.Figure, save_name: str):
    # save as png
    save_path = log_service.build_path('plots', save_name + '.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=120)

    plt.close(fig)


def _save_latex_plots(save_name: str):
    # TODO: Deprecated since it should be possible to convert PDF to EPS if necessary. EPS is inferior since it doesn't support transparency.
    # save as eps (good format for Latex)
    # save_path = log_service.build_path('plots', 'eps', save_name + '.eps')
    # plt.savefig(save_path, bbox_inches='tight', format='eps')

    # save as pdf
    save_path = log_service.build_path('plots', 'pdf', save_name + '.pdf')
    plt.savefig(save_path, bbox_inches='tight', dpi=120)


def save_and_finalize_plot(fig: plt.Figure, title: str, save_name: str):
    _save_latex_plots(save_name)
    plt.title(title)
    _save_and_close_plot(fig, save_name)


def plot_histogram(x: Sequence, y: Sequence, x_label: str, y_label: str, title: str, save_name: str, incline_labels=False):
    fig = plt.figure()
    plt.bar(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # add y grid lines
    plt.grid(b=True, which='both', axis='y', alpha=0.5, color='k')

    # use inclined x-labels
    if incline_labels:
        plt.gcf().autofmt_xdate()

    save_and_finalize_plot(fig, title, save_name)


def plot_multibar_histogram(x: Sequence, y_bars: 'list[BarInfo]', col_width: float, x_label: str, y_label: str, title: str, save_name: str):
    fig, ax = plt.subplots()

    x_label_dist = np.arange(len(x))  # the label locations
    x_offset = - ((len(y_bars) - 1) / 2.0) * col_width

    # Make the plot
    for bar_info in y_bars:
        ax.bar(x_label_dist + x_offset, bar_info.y, color=bar_info.color, width=col_width, label=bar_info.label)
        x_offset += col_width

    ax.set_xticks(x_label_dist)
    ax.set_xticklabels(x)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # add y grid lines
    plt.grid(b=True, which='both', axis='y', alpha=0.5, color='silver')

    ax.legend()

    save_and_finalize_plot(fig, title, save_name)


def plot_boxplot(values: Sequence, labels: Sequence[str], x_label: str, y_label: str, title: str, save_name: str, incline_labels=False):
    fig = plt.figure()
    plt.boxplot(values, labels=labels)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # add y grid lines
    plt.grid(b=True, which='both', axis='y', alpha=0.5, color='k')

    # use inclined x-labels
    if incline_labels:
        plt.gcf().autofmt_xdate()

    save_and_finalize_plot(fig, title, save_name)


def plot_pie_chart(values: Sequence, labels: Sequence[str], title: str, save_name: str):
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

    save_and_finalize_plot(fig, title, save_name)


def plot_scatter(x: Sequence, y: Sequence, x_label: str, y_label: str, title: str, save_name: str, legend_labels: Optional[Sequence[str]] = None):
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
        plt.scatter(x, y, marker='.', label=legend_labels[0])

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.legend(fontsize='x-small')

    save_and_finalize_plot(fig, title, save_name)


def plot_squared_scatter_chart(x: Sequence, y: Sequence, x_label: str, y_label: str, title: str, save_name: str = None,
                               plot_reference: bool = True, legend_labels: Optional[Sequence[str]] = None, value_range: Optional[tuple] = None):
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
        plt.scatter(x, y, marker='.', label=legend_labels[0])

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

    if save_name is not None:
        save_and_finalize_plot(fig, title, save_name)
    else:
        return fig


def plot_pareto_front(real_coords: Collection[list], pred_coords: Collection[list], labels: 'list[str]', title: str, save_name: str):
    ''' Plot Pareto front in 3D plot. If only 2 objectives are used in the Pareto optimization, rank is added as x-axis. '''
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')   # type: plt.Axes

    if len(real_coords) != len(pred_coords):
        raise ValueError(f'Inconsistent coordinates, real are {len(real_coords)}D while pred are {len(pred_coords)}D')
    if len(real_coords) != len(labels):
        raise ValueError('Labels count missmatch with coordinates dimensions')

    # already in 3D case, simply plot it
    if len(real_coords) == 3:
        ax.plot(*real_coords, '.b', alpha=0.6)
        ax.plot(*pred_coords, '.g', alpha=0.6)
        # ax.plot_wireframe(*real_coords, color='b', alpha=0.6)
        # ax.plot_wireframe(*pred_coords, color='g', alpha=0.6)

        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])
    # For 2 metrics only, adding another axis for displaying the rank, resulting in a 3D plot.
    elif len(real_coords) == 2:
        x_real, y_real = real_coords
        x_pred, y_pred = pred_coords

        x_real.reverse()
        y_real.reverse()
        x_pred.reverse()
        y_pred.reverse()

        ax.plot(list(range(len(x_real))), y_real, x_real, '--.b', alpha=0.6)
        ax.plot(list(range(len(x_pred))), y_pred, x_pred, '--.g', alpha=0.6)

        ax.set_xlabel('rank')
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[0])
    else:
        raise ValueError('Unsupported coordinate dimension')

    save_and_finalize_plot(fig, title, save_name)


def plot_predictions_pareto_scatter_chart(predictions: Sequence[list], pareto_points: Sequence[list],
                                          labels: 'list[str]', title: str, save_name: str):
    ''' Plot all predictions made, with the Pareto selected as highlight. '''
    # fig, ax = plt.subplots()
    fig = plt.figure()
    # rectilinear is for 2d
    proj = '3d' if len(predictions) >= 3 else 'rectilinear'

    ax = fig.add_subplot(projection=proj)  # type: plt.Axes

    point_dim = (72.*1.5/fig.dpi)**2
    num_pred_points = len(predictions[0])
    # in case of huge amount of predictions (standard case in NAS), if >= 0.5% of the points overlaps in a point saturate the alpha,
    # otherwise it is partially transparent. Clamped to 0.0125 since minimum is 1 / 255 = 0.004, but it's very hard to spot on plot.
    alpha = 0.25 if num_pred_points < 1000 else max(0.0125, 1 / (num_pred_points * 0.005))

    # zorder is used for plotting series over others (highest has priority)
    if proj == '3d':
        ax.scatter(*predictions, marker='o', alpha=alpha, zorder=1, s=point_dim, rasterized=True)
        ax.scatter(*pareto_points, marker='*', alpha=1.0, zorder=2)

        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])
    else:
        # accuracy is put first, but it's better to have it on y-axis for readability
        ax.scatter(*reversed(predictions), marker='o', alpha=alpha, zorder=1, s=point_dim, rasterized=True)
        ax.plot(*reversed(pareto_points), '*--', color='tab:orange', alpha=1.0, zorder=2)

        ax.set_xlabel(labels[1])
        ax.set_ylabel(labels[0])

    save_and_finalize_plot(fig, title, save_name)
