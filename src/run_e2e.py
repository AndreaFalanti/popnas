import argparse
import os.path
import sys

import pandas as pd

import log_service
import scripts
from popnas import Popnas
from utils.config_dataclasses import RunConfig
from utils.config_utils import validate_config_json, initialize_search_config_and_logs
from utils.func_utils import run_as_sequential_process
from utils.nn_utils import initialize_train_strategy


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-r', metavar='RESTORE_FOLDER', type=str, help='path of log folder to restore', default=None)
    parser.add_argument('-j', metavar='JSON_PATH', type=str, help='path to config json for NAS', default=os.path.join('configs', 'run.json'))
    parser.add_argument('-jms', metavar='JSON_PATH_MODEL_SEARCH', type=str, help='path to config json for model search', default=None)
    parser.add_argument('-jlt', metavar='JSON_PATH', type=str, help='path to config json for last training', default=None)
    parser.add_argument('-k', metavar='NUM_TOP_MODELS', type=int, help='number of top models to consider in model selection', default=5)
    parser.add_argument('-params', metavar='PARAMS RANGE', type=str,
                        help="desired params range for model selection, semicolon separated (e.g. 2.5e6;3.5e6)", default=None)
    parser.add_argument('-b', metavar='POST_SEARCH_BATCH_SIZE', type=int, help='batch size used in post search scripts', default=None)
    parser.add_argument('-name', metavar='RUN_NAME', type=str, help='name used for log folder', default=None)
    args = parser.parse_args()

    # create folder structure for log files or reuse a previous log folder to continue a stopped/crashed execution
    run_config = initialize_search_config_and_logs(args.name, args.j, args.r)
    # handle uncaught exception in a special log file
    # leave it before validating JSON so that the config exception is logged correctly when triggered
    sys.excepthook = log_service.make_exception_handler(log_service.create_critical_logger())
    # check that the config is correct
    validate_config_json(run_config)

    log_path = log_service.log_path
    post_batch_size = run_config.dataset.batch_size if args.b is None else args.b

    run_as_sequential_process(f=run_search, args=(log_path, run_config))
    run_as_sequential_process(f=scripts.execute_model_selection_training,
                              kwargs={'p': log_path, 'b': post_batch_size, 'params': args.params, 'j': args.jms, 'k': args.k})

    # get the best result found during model selection
    model_selection_results_csv_path = os.path.join(log_path, f'best_model_training_top{args.k}', 'training_results.csv')
    ms_df = pd.read_csv(model_selection_results_csv_path)
    best_ms = ms_df[ms_df['val_score'] == ms_df['val_score'].max()].to_dict('records')[0]
    print(f'Best model found during model selection: {best_ms}')

    run_as_sequential_process(f=scripts.execute_last_training,
                              kwargs={'p': log_path, 'b': post_batch_size, 'f': best_ms['f'], 'm': best_ms['m'], 'n': best_ms['n'],
                                      'spec': best_ms['cell_spec'], 'j': args.jlt})


def run_search(log_path: str, run_config: RunConfig):
    log_service.set_log_path(log_path)
    train_strategy = initialize_train_strategy(run_config.others.train_strategy, run_config.others.use_mixed_precision)

    popnas = Popnas(run_config, train_strategy)
    popnas.start()
    scripts.generate_plot_slides(log_path, save=True)


if __name__ == '__main__':
    main()
