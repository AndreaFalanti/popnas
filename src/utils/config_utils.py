import dataclasses
import json
import os

import jsonschema
from dacite import from_dict
from neptune.utils import stringify_unsupported

import log_service
from utils.config_dataclasses import RunConfig


def read_json_config(config_path: str) -> RunConfig:
    with open(config_path, 'r') as f:
        data = json.load(f)
        return from_dict(data_class=RunConfig, data=data)


def validate_config_json(config: RunConfig):
    with open('config_schema.json') as f:
        schema = json.load(f)

    # validate JSON structure and values
    jsonschema.validate(dataclasses.asdict(config), schema)

    # extra logic to check consistencies between multiple fields
    if not config.others.pnas_mode and len(config.search_strategy.additional_pareto_objectives) == 0:
        raise ValueError('POPNAS mode requires at least two Pareto objectives to optimize (score + at least 1 additional objective)')

    if config.others.pnas_mode:
        config.search_strategy.additional_pareto_objectives = []


def initialize_search_config_and_logs(log_folder_name: str, json_config_path: str, restore_path: str) -> RunConfig:
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
        run_config = read_json_config(log_service.build_path('restore', 'run.json'))
    else:
        log_service.initialize_log_folders(log_folder_name)
        run_config = read_json_config(json_config_path)

        # copy config for possible run restore and post-search scripts
        with open(log_service.build_path('restore', 'run.json'), 'w') as f:
            config_dict = dataclasses.asdict(run_config)
            json.dump(config_dict, f, indent=4)
            if log_service.neptune_project is not None:
                log_service.neptune_project['popnas_config'] = stringify_unsupported(config_dict)

    return run_config


def retrieve_search_config(log_folder_path: str) -> RunConfig:
    run_config_path = os.path.join(log_folder_path, 'restore', 'run.json')
    return read_json_config(run_config_path)
