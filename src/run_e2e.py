import argparse
import json
import os.path
import sys

import pandas as pd

import log_service
import scripts
from popnas import Popnas
from utils.config_validator import validate_config_json
from utils.nn_utils import initialize_train_strategy


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-r', metavar='RESTORE_FOLDER', type=str, help='path of log folder to restore', default=None)
    parser.add_argument('-j', metavar='JSON_PATH', type=str, help='path to config json for NAS', default=None)
    parser.add_argument('-jms', metavar='JSON_PATH_MODEL_SEARCH', type=str, help='path to config json for model search', default=None)
    parser.add_argument('-jlt', metavar='JSON_PATH', type=str, help='path to config json for last training', default=None)
    parser.add_argument('-params', metavar='PARAMS RANGE', type=str,
                        help="desired params range for model selection, semicolon separated (e.g. 2.5e6;3.5e6)", default=None)
    parser.add_argument('-b', metavar='POST_SEARCH_BATCH_SIZE', type=int, help='batch size used in post search scripts', default=None)
    parser.add_argument('-name', metavar='RUN_NAME', type=str, help='name used for log folder', default=None)
    args = parser.parse_args()

    # create folder structure for log files or reuse previous logs to continue execution
    if args.r is not None:
        log_service.check_log_folder(args.r)
        # load the exact configuration in which the run was started (also setting CPU/GPU properly to maintain training time consistency)
        with open(log_service.build_path('restore', 'run.json'), 'r') as f:
            run_config = json.load(f)
    else:
        log_service.initialize_log_folders(args.name)

        json_path = os.path.join('configs', 'run.json') if args.j is None else args.j
        with open(json_path, 'r') as f:
            run_config = json.load(f)

        # copy config (with args override) for possible run restore
        with open(log_service.build_path('restore', 'run.json'), 'w') as f:
            json.dump(run_config, f, indent=4)

    # Handle uncaught exception in a special log file
    sys.excepthook = log_service.make_exception_handler(log_service.create_critical_logger())

    # check that the config is correct
    validate_config_json(run_config)

    # DEBUG: To find out which devices your operations and tensors are assigned to
    # tf.debugging.set_log_device_placement(True)

    train_strategy = initialize_train_strategy(run_config['others']['train_strategy'])

    popnas = Popnas(run_config, train_strategy)
    popnas.start()

    run_folder_path = log_service.log_path
    post_batch_size = run_config['dataset']['batch_size'] if args.b is None else args.b
    scripts.generate_plot_slides(run_folder_path, save=True)
    scripts.execute_model_selection_training(run_folder_path, b=post_batch_size, params=args.params, j=args.jms)

    # get model selection best result
    model_selection_results_csv_path = os.path.join(run_folder_path, 'best_model_training_top5', 'training_results.csv')
    ms_df = pd.read_csv(model_selection_results_csv_path)
    best_ms = ms_df[ms_df['val_score'] == ms_df['val_score'].max()].to_dict('records')[0]
    print(f'Best model found: {best_ms}')

    scripts.execute_last_training(run_folder_path, b=post_batch_size, f=best_ms['f'], m=best_ms['m'], n=best_ms['n'], spec=best_ms['cell_spec'],
                                  j=args.jlt)


if __name__ == '__main__':
    main()
