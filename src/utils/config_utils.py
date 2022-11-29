import json
import os
from typing import Any

import jsonschema

import log_service


def validate_config_json(config: dict):
    with open('config_schema.json') as f:
        schema = json.load(f)

    # validate JSON structure and values
    jsonschema.validate(config, schema)

    # extra logic to check consistencies between multiple fields
    if not config['others']['pnas_mode'] and len(config['search_strategy']['additional_pareto_objectives']) == 0:
        raise ValueError('POPNAS mode requires at least two Pareto objectives to optimize (score + at least 1 additional objective')

    if config['others']['pnas_mode']:
        config['search_strategy']['additional_pareto_objectives'] = []


def initialize_search_config_and_logs(log_folder_name: str, json_config_path: str, restore_path: str) -> 'dict[str, Any]':
    '''
    Initialize the log folders and read the configuration from the JSON path provided.
    If restore_path is specified, the folder is set to that path and the config restored from that, ignoring the first two arguments.

    Args:
        log_folder_name: name for log folder associated to the run
        json_config_path: path to JSON configuration file used by POPNAS
        restore_path: path to restore, provided only when resuming a previous run

    Returns:
        the NAS run configuration
    '''
    if restore_path is not None:
        log_service.restore_log_folder(restore_path)
        # load the exact configuration provided when the run was started
        with open(log_service.build_path('restore', 'run.json'), 'r') as f:
            run_config = json.load(f)
    else:
        log_service.initialize_log_folders(log_folder_name)

        json_path = os.path.join('configs', 'run.json') if json_config_path is None else json_config_path
        with open(json_path, 'r') as f:
            run_config = json.load(f)

        # copy config for possible run restore and post-search scripts
        with open(log_service.build_path('restore', 'run.json'), 'w') as f:
            json.dump(run_config, f, indent=4)

    return run_config
