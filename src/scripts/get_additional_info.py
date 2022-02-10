import argparse
import os.path
import re

import pandas as pd


def get_total_networks_pruned_by_equivalence(log_file_path: str):
    ''' Returns the total count of networks pruned by the equivalence check. It is done by simply reading the debug log. '''
    pruned_cells_count = 0

    with open(log_file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        match = re.search(r'Pruned (\d+)', line)
        if match:
            pruned_cells_count += int(match.group(1))

    return pruned_cells_count


def get_total_networks_introduced_by_exploration(training_results_path: str):
    ''' Returns the total amount of cell expansions that use as base structure an architecture selected in the exploration step. '''
    results_df = pd.read_csv(training_results_path)
    # remove square brackets to make easier the cell comparison
    results_df['cell structure'] = results_df['cell structure'].map(lambda val: re.sub(r'[\[\]]', '', val))

    exploration_cells = results_df[results_df['exploration'] == True]['cell structure'].to_list()  # type: list['str']
    standard_cells = results_df[results_df['exploration'] == False]['cell structure'].to_list()  # type: list['str']

    exploration_based_cells = [cell for cell in standard_cells if any([cell.startswith(exp_cell) for exp_cell in exploration_cells])]

    return len(exploration_based_cells)


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-p', metavar='FOLDER', type=str, help="log folder", required=True)
    args = parser.parse_args()

    log_file_path = os.path.join(args.p, 'debug.log')
    results_csv_path = os.path.join(args.p, 'csv', 'training_results.csv')

    eqv_pruned_cells = get_total_networks_pruned_by_equivalence(log_file_path)
    exp_induced_cells = get_total_networks_introduced_by_exploration(results_csv_path)

    # write these results in a new file
    write_path = os.path.join(args.p, 'extras.txt')
    with open(write_path, 'w') as f:
        f.write('Total cells pruned by equivalence check: ' + str(eqv_pruned_cells))
        f.write('\nTotal cells introduced by exploration step: ' + str(exp_induced_cells))


if __name__ == '__main__':
    main()
