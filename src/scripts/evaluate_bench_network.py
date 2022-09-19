import argparse
import json
import os

import pandas as pd

import log_service
from benchmarking import NATSbench
from utils.func_utils import parse_cell_structures
from utils.rstr import rstr


def get_best_cell_spec(log_folder_path: str, metric: str = 'best val accuracy'):
    training_results_csv_path = os.path.join(log_folder_path, 'csv', 'training_results.csv')
    df = pd.read_csv(training_results_csv_path)
    best_acc_row = df.loc[df[metric].idxmax()]

    cell_spec = parse_cell_structures([best_acc_row['cell structure']])[0]
    return cell_spec, best_acc_row[metric]


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-p', metavar='PATH', type=str, help="path to log folder", required=True)
    parser.add_argument('-f', metavar='JSON_PATH', type=str, help='path to bench files folder', default=None, required=True)
    args = parser.parse_args()

    config_json_path = os.path.join(args.p, 'restore', 'run.json')
    print('Reading configuration...')
    with open(config_json_path, 'r') as f:
        config = json.load(f)

    log_service.set_log_path(args.p)
    # Create the API instance for the topology search space in NATS
    bench = NATSbench(args.f)

    best_cell_spec, best_val = get_best_cell_spec(args.p)
    dataset = config['dataset']['name']

    test_acc = bench.simulate_testing_on_nas_bench_201(best_cell_spec, dataset)

    print(f'Test accuracy: {test_acc}')
    with open(os.path.join(args.p, 'bench-test.txt'), 'w') as f:
        f.write(f'Cell: {rstr(best_cell_spec)}\n')
        f.write(f'val: {best_val}, test: {test_acc}')


if __name__ == '__main__':
    main()
