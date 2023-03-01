import argparse
import dataclasses
import json
import math
import os.path
import pickle
import sys
from typing import Any

import pandas as pd
from dacite import from_dict

import log_service
import scripts
from popnas import Popnas
from utils.config_dataclasses import RunConfig
from utils.config_utils import validate_config_json
from utils.func_utils import clamp, run_as_sequential_process
from utils.nn_utils import initialize_train_strategy
from utils.tsc_utils import TSCDatasetMetadata


def adapt_config_to_dataset_metadata(base_config: 'dict[str, Any]', ds_meta: TSCDatasetMetadata) -> RunConfig:
    # deep copy of base configuration
    # new_config_dict = json.loads(json.dumps(base_config))
    new_config = from_dict(data_class=RunConfig, data=base_config)

    new_config.dataset.name = ds_meta.name
    root_ds_path = 'multivariate' if ds_meta.multivariate else 'univariate'
    new_config.dataset.path = os.path.join('datasets', 'UCR-UEA-archives', root_ds_path, ds_meta.name)
    new_config.dataset.classes_count = ds_meta.classes

    # adapt validation size
    val_size = new_config.dataset.validation_size
    # have at least 60 samples for validation, otherwise validation accuracy can become really noisy.
    # as example, if you have 20 samples, you have 5% accuracy gaps between when an architecture predicts just one sample better than another one!
    # accuracy predictor can't reliably predict the quality on such a low number of samples.
    val_min_samples = 60
    if val_size is not None and val_size * ds_meta.train_size < val_min_samples:
        # 100 factors are to keep it as a .2f fraction
        val_size = math.ceil((val_min_samples * 100) / ds_meta.train_size) / 100

    # address a rare case where the validation set could have fewer samples than classes, causing an error during the train-val split
    if val_size is not None and val_size * ds_meta.train_size < ds_meta.classes:
        val_size = min([0.10 + i * 0.05 for i in range(8) if (0.10 + i * 0.05) * ds_meta.train_size >= ds_meta.classes])

    # override val_size with potential new value
    new_config.dataset.validation_size = val_size

    # adapt batch size to have at least 10 train batches per epoch
    val_size = 0 if val_size is None else val_size
    train_split_size = 1 - val_size
    # batch size scales by 16: min value is 16, max is 128.
    batch_units = 16
    batch_mult = int(clamp((ds_meta.train_size * train_split_size) // (10 * batch_units), 1, 8))
    new_config.dataset.batch_size = batch_units * batch_mult

    # TODO: could be a good idea to adapt also motifs, based on the number of timesteps.
    #  Still, model selection can take care of that but more reduction cells can help for long time series even in search.

    return new_config


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-p', metavar='PICKLE_FILE', type=str, help='path of pickle file containing the metadata of the datasets to process',
                        required=True)
    parser.add_argument('-name', metavar='RUN_NAME', type=str, help='name used for log folder', required=True)
    parser.add_argument('-j', metavar='JSON_PATH', type=str, help='path to config json for NAS', default=None)
    parser.add_argument('-jms', metavar='JSON_PATH_MODEL_SEARCH', type=str, help='path to config json for model search', default=None)
    parser.add_argument('-jlt', metavar='JSON_PATH', type=str, help='path to config json for last training', default=None)
    parser.add_argument('-params', metavar='PARAMS RANGE', type=str,
                        help="desired params range for model selection, semicolon separated (e.g. 2.5e6;3.5e6)", default=None)
    parser.add_argument('-b', metavar='POST_SEARCH_BATCH_SIZE', type=int, help='batch size used in post search scripts', default=None)
    args = parser.parse_args()

    with open(args.p, 'rb') as f:
        datasets_metadata = pickle.load(f)  # type: list[TSCDatasetMetadata]

    ds_names = [ds_meta.name for ds_meta in datasets_metadata]
    print(f'This worker will process {len(ds_names)} datasets: {ds_names}')

    try:
        os.makedirs(os.path.join('logs', args.name))
    except OSError:
        print('Root log folder already present. This worker will insert the experiments about the new datasets processed to that run.')

    print('Reading base configuration')
    json_path = os.path.join('configs', 'run_ts.json') if args.j is None else args.j
    with open(json_path, 'r') as f:
        base_config = json.load(f)

    for ds_meta in datasets_metadata:
        # create folder structure for log files
        root_folder = 'multivariate' if ds_meta.multivariate else 'univariate'
        run_folder_path = os.path.join(args.name, root_folder, ds_meta.name)
        try:
            log_service.initialize_log_folders(run_folder_path)
        except AttributeError:
            print(f'Skipping dataset "{ds_meta.name}" since its folder already exists...')
            continue

        log_path = os.path.join('logs', run_folder_path)
        customized_config = adapt_config_to_dataset_metadata(base_config, ds_meta)
        post_batch_size = customized_config.dataset.batch_size if args.b is None else args.b

        # NOTE: run each step of the E2E experiment in a separate process, to "encapsulate and destroy" memory leaks caused by Tensorflow...
        # processes are run sequentially (immediate join). The workflow is exactly the same as just calling the functions.
        run_as_sequential_process(f=run_search_experiment, args=(log_path, customized_config))
        run_as_sequential_process(f=scripts.execute_model_selection_training,
                                  kwargs={'p': log_path, 'b': post_batch_size, 'params': args.params, 'j': args.jms})
        run_as_sequential_process(f=run_last_training, args=(log_path, post_batch_size, args.jlt))


def run_search_experiment(log_path: str, customized_config: RunConfig):
    log_service.set_log_path(log_path)

    # copy config for possible run restore and post-search scripts
    with open(log_service.build_path('restore', 'run.json'), 'w') as f:
        json.dump(dataclasses.asdict(customized_config), f, indent=4)

    # handle uncaught exception in a special log file
    # leave it before validating JSON so that the config exception is logged correctly when triggered
    sys.excepthook = log_service.make_exception_handler(log_service.create_critical_logger())
    # check that the config is correct
    validate_config_json(customized_config)

    train_strategy = initialize_train_strategy(customized_config.others.train_strategy)

    popnas = Popnas(customized_config, train_strategy)
    popnas.start()

    scripts.generate_plot_slides(log_path, save=True)


def run_last_training(run_folder_path: str, batch_size: int, json_path: str):
    # get the best result found during model selection
    model_selection_results_csv_path = os.path.join(run_folder_path, 'best_model_training_top5', 'training_results.csv')
    ms_df = pd.read_csv(model_selection_results_csv_path)
    best_ms = ms_df[ms_df['val_score'] == ms_df['val_score'].max()].to_dict('records')[0]
    print(f'Best model found during model selection: {best_ms}')

    scripts.execute_last_training(run_folder_path, b=batch_size, f=best_ms['f'], m=best_ms['m'], n=best_ms['n'], spec=best_ms['cell_spec'],
                                  j=json_path)


if __name__ == '__main__':
    main()
