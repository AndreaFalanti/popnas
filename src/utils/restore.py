import os
import pickle
from typing import Type

import pandas as pd

import log_service
from models.results import BaseTrainingResults
from search_space import SearchSpace
from search_space_units import parse_cell_strings


class RestoreInfo:
    def __init__(self, save_path: str) -> None:
        # TODO: shouldn't be in class but log_service must be initialized...
        self.save_path = save_path

        self._current_b = 0
        self._pareto_training_index = 0
        self._exploration_training_index = 0
        self._total_time = 0

    # TODO: using **kwargs can be useful here, but names become error prone in that case. Instead use same names of actual arguments.
    def update(self, current_b: int = None, pareto_training_index: int = None, exploration_training_index: int = None, total_time: float = None):
        # get dictionary of parameters-values passed to the function, with also the defaults
        args = locals().copy()
        del args['self']

        for key, val in args.items():
            if val is not None:
                setattr(self, f'_{key}', val)

        with open(self.save_path, 'wb') as f:
            pickle.dump(self, f)

    def get_info(self):
        return {
            'current_b': self._current_b,
            'pareto_training_index': self._pareto_training_index,
            'exploration_training_index': self._exploration_training_index,
            'total_time': self._total_time
        }

    def must_restore_dynamic_reindex_function(self):
        return self._current_b >= 2

    def must_restore_search_space_children(self):
        return self._current_b >= 2


def restore_dynamic_reindex_function():
    empty_cell_data = pd.read_csv(log_service.build_path('csv', 'training_results.csv'), nrows=1)

    smb_data = pd.read_csv(log_service.build_path('csv', 'reindex_op_times.csv'), names=['time', 'op'])
    op_times = {row.op: row.time for row in smb_data.itertuples(index=False)}

    return op_times, empty_cell_data['training time(seconds)'][0]


def restore_inference_dynamic_reindex_function():
    empty_cell_data = pd.read_csv(log_service.build_path('csv', 'training_results.csv'), nrows=1)

    smb_data = pd.read_csv(log_service.build_path('csv', 'reindex_op_times.csv'), names=['inf_time', 'op'])
    op_times = {row.op: row.inf_time for row in smb_data.itertuples(index=False)}

    return op_times, empty_cell_data['inference time(seconds)'][0]


def restore_train_info(b: int, results_class: Type[BaseTrainingResults]):
    training_df = pd.read_csv(log_service.build_path('csv', 'training_results.csv'))
    # filter on current block level
    training_df = training_df[training_df['# blocks'] == b].drop(columns=['exploration'])
    training_df['cell structure'] = parse_cell_strings(training_df['cell structure'])

    return [results_class.from_csv_row(row) for row in training_df.itertuples(index=False)]


def restore_search_space_children(search_space: SearchSpace, b: int, max_children: int, pnas_mode: bool):
    children_csv_filename = f'predictions_B{b}.csv' if pnas_mode else f'pareto_front_B{b}.csv'
    children_df = pd.read_csv(log_service.build_path('csv', children_csv_filename)).head(max_children)
    search_space.children = parse_cell_strings(children_df['cell structure'])

    exploration_csv_path = log_service.build_path('csv', f'exploration_pareto_front_B{b}.csv')
    if os.path.exists(exploration_csv_path):
        exploration_df = pd.read_csv(exploration_csv_path)
        search_space.exploration_front = parse_cell_strings(exploration_df['cell structure'])
    else:
        search_space.exploration_front = []
