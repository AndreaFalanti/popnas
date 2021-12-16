import argparse
import os

import pandas as pd

import log_service
from utils.feature_utils import *
from utils.func_utils import parse_cell_structures, instantiate_search_space_from_logs


def write_new_time_dataset_csv(cells_info: 'list[tuple[list, float, int]]', search_space: SearchSpace):
    # headers, header_types = build_feature_names(label_col, max_blocks, max_lookback)
    headers = ['time'] + ['blocks', 'cells', 'op_score', 'multiple_lookbacks', 'first_dag_level_op_score_fraction', 'block_dependencies',
                          'dag_depth', 'concat_inputs', 'heaviest_dag_path_op_score_fraction'] + ['exploration']
    header_types = ['Label'] + ['Num'] * 9 + ['Auxiliary']

    csv_rows = []

    # address initial thrust separately
    _, target, _ = cells_info[0]
    csv_rows.append([target] + [0, 0, 0, 0, 1, 0, 0, 0, 1] + [False])

    for cell_spec, target, blocks in cells_info[1:]:
        time_features = generate_time_features(cell_spec, search_space)

        features_list = [target] + time_features + [False]
        csv_rows.append(features_list)

    with open(log_service.build_path('csv', f'new_training_time.csv'), mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(csv_rows)

    with open(log_service.build_path('csv', f'new_column_desc_time.csv'), mode='w', newline='') as f:
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

    # get some search space info from the run logs, so that SearchSpace instance can be correctly initialized
    search_space = instantiate_search_space_from_logs(args.p)

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

    search_space.add_operator_encoder('dynamic_reindex', fn=reindex_function)

    print('Generating new datasets...')
    write_new_time_dataset_csv(list(zip(cells, time_list, blocks_list)), search_space)
    # write_new_dataset_csv('acc', list(zip(cells, acc_list, blocks_list)), search_space)


if __name__ == '__main__':
    main()
