import json

import jsonschema


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

