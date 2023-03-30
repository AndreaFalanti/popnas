import argparse
import os
import sys

import log_service
from popnas import Popnas
from utils.config_utils import validate_config_json, initialize_search_config_and_logs
from utils.nn_utils import initialize_train_strategy


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-r', metavar='RESTORE_FOLDER', type=str, help='path of log folder to restore', default=None)
    parser.add_argument('-j', metavar='JSON_PATH', type=str, help='path to config json with run parameters',
                        default=os.path.join('configs', 'run.json'))
    parser.add_argument('--name', metavar='RUN_NAME', type=str, help='name used for log folder', default=None)
    args = parser.parse_args()

    # create folder structure for log files or reuse a previous log folder to continue a stopped/crashed execution
    run_config = initialize_search_config_and_logs(args.name, args.j, args.r)
    # handle uncaught exception in a special log file
    # leave it before validating JSON so that the config exception is logged correctly when triggered
    sys.excepthook = log_service.make_exception_handler(log_service.create_critical_logger())
    # check that the config is correct
    validate_config_json(run_config)

    train_strategy = initialize_train_strategy(run_config.others.train_strategy, run_config.others.use_mixed_precision)

    popnas = Popnas(run_config, train_strategy)
    popnas.start()


if __name__ == '__main__':
    main()
