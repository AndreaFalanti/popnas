import argparse
import csv
import os

import pandas as pd

import log_service
from encoder import StateSpace
from utils.feature_utils import *
from utils.func_utils import parse_cell_structures


def write_new_dataset_csv(label_col: str, cells_info: 'list[tuple[list, float, int]]', state_space: StateSpace, max_cells: int):
    max_blocks = state_space.B
    max_lookback = abs(state_space.input_lookback_depth)

    headers, header_types = build_feature_names(label_col, max_blocks, max_lookback)

    csv_rows = []

    # address initial thrust separately
    _, target, blocks = cells_info[0]
    csv_rows.append([target, blocks, 0] + [0] * (len(headers) - 4) + [False])

    for cell_spec, target, blocks in cells_info[1:]:
        # equivalent cells can be useful to train better the regressor
        eqv_cells, _ = state_space.generate_eqv_cells(cell_spec, size=max_blocks)

        # expand cell_spec for bool comparison of data_augmented field
        cell_spec = cell_spec + [(None, None, None, None)] * (max_blocks - blocks)

        for eqv_cell in eqv_cells:
            total_cells = compute_real_cnn_cell_stack_depth(eqv_cell, max_cells)
            lookback_usage_features = compute_lookback_usage_features(eqv_cell, max_lookback)
            lookback_incidence_features = compute_blocks_lookback_incidence_matrix(eqv_cell, max_blocks, max_lookback)
            block_incidence_features = compute_blocks_incidence_matrix(eqv_cell, max_blocks)

            op_enc = 'dynamic_reindex' if label_col == 'time' else 'cat'
            op_features = state_space.encode_cell_spec(eqv_cell, op_enc_name=op_enc)[1::2]

            features_list = [target, blocks, total_cells] + op_features + lookback_usage_features + \
                            lookback_incidence_features + block_incidence_features + [eqv_cell != cell_spec]
            csv_rows.append(features_list)

    with open(log_service.build_path('csv', f'new_training_{label_col}.csv'), mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(csv_rows)

    with open(log_service.build_path('csv', f'new_column_desc_{label_col}.csv'), mode='w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(enumerate(header_types))


def read_training_results(file_path: str):
    training_df = pd.read_csv(file_path)

    parsed_cells = parse_cell_structures(training_df['cell structure'])

    return parsed_cells, training_df['training time(seconds)'].to_list(), \
           training_df['best val accuracy'].to_list(), training_df['# blocks'].to_list()


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-p', metavar='FOLDER', type=str, help="log folder", required=True)
    args = parser.parse_args()

    log_service.set_log_path(os.path.join(args.p))

    operators = ['identity', '3x3 dconv', '5x5 dconv', '7x7 dconv', '1x7-7x1 conv', '3x3 conv', '3x3 maxpool', '3x3 avgpool']
    state_space = StateSpace(5, operators, 8, input_lookback_depth=-2)

    print('Processing old dataset...')
    old_dataset_file_path = os.path.join(args.p, 'csv', 'training_results.csv')
    cells, time_list, acc_list, blocks_list = read_training_results(old_dataset_file_path)

    print('Generating reindex function...')
    reindex_file_path = os.path.join(args.p, 'csv', 'reindex_op_times.csv')
    reindex_df = pd.read_csv(reindex_file_path, names=['time', 'op'])

    # abs is used because identity takes about the same time of GAP, so the variance in processing speed can produce small negative values.
    # assume instead that it takes a little more time, because it actually make no sense to have a negative value.
    times_without_bias = [abs(time - time_list[0]) for time in reindex_df['time'].to_list()]

    reindex_dict = {}
    max_time = max(times_without_bias)
    for op, time in zip(reindex_df['op'].to_list(), times_without_bias):
        reindex_dict[op] = time / max_time

    def reindex_function(op_value: str):
        return reindex_dict[op_value]

    state_space.add_operator_encoder('dynamic_reindex', fn=reindex_function)

    print('Generating new datasets...')
    write_new_dataset_csv('time', list(zip(cells, time_list, blocks_list)), state_space, max_cells=5)
    write_new_dataset_csv('acc', list(zip(cells, acc_list, blocks_list)), state_space, max_cells=5)


if __name__ == '__main__':
    main()
