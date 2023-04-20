'''
Stores common function used in post search scripts, which could be executed after the main NAS procedure.
All the functions stored inside this file are not used in the search algorithm.
'''
import copy
import dataclasses
import json
import logging
import os
from typing import NamedTuple, Optional

import pandas as pd
import tensorflow as tf
from dacite import from_dict
from mergedeep import merge
from tensorflow.keras import callbacks, Model, losses, metrics

import log_service
from models.custom_callbacks import ModelCheckpointCustom
from models.generators.base import BaseModelGenerator
from models.results.base import TargetMetric, get_best_metric_and_epoch_index
from search_space import CellSpecification, parse_cell_strings
from utils.config_dataclasses import RunConfig
from utils.func_utils import create_empty_folder, from_seconds_to_hms
from utils.nn_utils import initialize_train_strategy, get_optimized_steps_per_execution


class MacroConfig(NamedTuple):
    m: int
    n: int
    f: int

    def __str__(self) -> str:
        return f'm{self.m}-n{self.n}-f{self.f}'

    def modify(self, m_mod: int, n_mod: int, f_mod: float):
        return MacroConfig(self.m + m_mod, self.n + n_mod, int(self.f * f_mod))

    @staticmethod
    def from_config(config: RunConfig):
        return MacroConfig(config.architecture_hyperparameters.motifs,
                           config.architecture_hyperparameters.normal_cells_per_motif,
                           config.architecture_hyperparameters.filters)


def define_callbacks(score_metric: str, output_names: 'list[str]', save_chunk: int = 100, use_val: bool = False) -> 'list[callbacks.Callback]':
    ''' Define callbacks used during model training in post search procedures. '''
    # Save best weights, considering the score metric of last output
    prefix = 'val_' if use_val else ''
    target_metric = f'{prefix}{output_names[-1]}_{score_metric}'

    ckpt_save_format = 'cp_ed{epoch:02d}_' + str(save_chunk) + '.ckpt'
    # TODO: best score metric could be the min, should use the .optimal value from the TargetMetric. Refactor this later...
    ckpt_callback = ModelCheckpointCustom(filepath=log_service.build_path('weights', ckpt_save_format),
                                          save_weights_only=True, save_best_only=True,
                                          monitor=target_metric, mode='max', save_chunk=save_chunk)
    # By default, it shows losses and metrics for both training and validation
    tb_callback = callbacks.TensorBoard(log_dir=log_service.build_path('tensorboard'), profile_batch=0, histogram_freq=0, write_graph=False)

    return [ckpt_callback, tb_callback]


def create_model_log_folder(log_path: str):
    create_empty_folder(log_path)
    os.mkdir(os.path.join(log_path, 'weights'))  # create weights folder
    os.mkdir(os.path.join(log_path, 'tensorboard'))  # create tensorboard folder

    log_service.set_log_path(log_path)


def dump_json_config(config: RunConfig, save_path: str):
    with open(os.path.join(save_path, 'run.json'), 'w') as f:
        json.dump(dataclasses.asdict(config), f, indent=4)


def extract_final_training_results(hist: callbacks.History, score_metric_name: str, keras_metrics: 'list[TargetMetric]',
                                   output_names: 'list[str]', using_val: bool = True):
    history = hist.history
    score_metric = next(m for m in keras_metrics if m.name == score_metric_name)

    epoch_index, best_score = get_best_metric_and_epoch_index(history, score_metric, output_names, using_val)

    best_epoch_results = {output_name: {m.name: history[m.to_keras_history_key(False, output_name)][epoch_index] for m in keras_metrics}
                          for output_name in output_names}
    if using_val:
        best_epoch_results.update(
            {output_name: {f'val_{m.name}': history[m.to_keras_history_key(True, output_name)][epoch_index] for m in keras_metrics}
             for output_name in output_names}
        )

    return best_epoch_results, epoch_index, best_score


def extend_keras_metrics(keras_metrics: 'list[TargetMetric]'):
    return [
        TargetMetric('loss', min, ''),
    ] + keras_metrics


def log_training_results_summary(logger: logging.Logger, best_epoch_index: int, total_epochs: int,
                                 training_time: float, best_score: float, score_metric_name: str):
    logger.info('*' * 31 + ' TRAINING SUMMARY ' + '*' * 31)
    logger.info('Best epoch index: %d', best_epoch_index + 1)
    logger.info('Total epochs: %d', total_epochs)
    logger.info('Best validation %s: %0.4f', score_metric_name, best_score)
    logger.info('Total training time (without callbacks): %0.3f seconds (%d hours %d minutes %d seconds)',
                training_time, *from_seconds_to_hms(training_time))


def log_training_results_dict(logger: logging.Logger, results_dict: 'dict[str, dict[str, float]]'):
    logger.info('*' * 30 + ' RESULTS PER OUTPUT ' + '*' * 30)

    for output_name, metrics_dict in results_dict.items():
        logger.info('Output: %s', output_name)
        for metric_name, value in metrics_dict.items():
            logger.info('\t%s: %0.4f', metric_name, value)

    logger.info('*' * 80)


def build_config(run_path: str, batch_size: int, train_strategy: str, custom_json_path: str) -> 'tuple[RunConfig, tf.distribute.Strategy]':
    # run configuration used during search
    with open(os.path.join(run_path, 'restore', 'run.json'), 'r') as f:
        search_config = json.load(f)

    # read hyperparameters to use for model selection, overriding search ones
    with open(custom_json_path, 'r') as f:
        ms_partial_config = json.load(f)  # type: dict

    ms_config_dict = merge({}, search_config, ms_partial_config)
    ms_config = from_dict(data_class=RunConfig, data=ms_config_dict)

    # force models to be multi-output (all post-search models have a secondary exit for improving generalization)
    ms_config.architecture_hyperparameters.multi_output = True

    # set custom batch size, if present
    if batch_size is not None:
        ms_config.training_hyperparameters.learning_rate = ms_config.training_hyperparameters.learning_rate * \
                                                           (batch_size / ms_config.dataset.batch_size)
        ms_config.dataset.batch_size = batch_size

    # initialize train strategy (try-except to be retrocompatible with the previous config format)
    try:
        if train_strategy is not None:
            ms_config.others.train_strategy = train_strategy
        train_strategy = initialize_train_strategy(ms_config.others.train_strategy, ms_config.others.use_mixed_precision)
    except AttributeError:
        train_strategy = initialize_train_strategy(None, False)

    return ms_config, train_strategy


def prune_excessive_outputs(mo_model: Model, mo_losses: 'dict[str, losses.Loss]', mo_loss_weights: 'dict[str, float]'):
    '''
    Build a new model using only a secondary output at 2/3 of cells (drop other outputs from multi-output model).
    Args:
        mo_model: a multi-output model
        mo_losses: dictionary associating an output name to a loss
        mo_loss_weights: loss weights associated with each output

    Returns:
        the new model, the new losses, the new loss weights and the new output names
    '''
    last_output_index = len(mo_model.outputs) - 1
    secondary_output_index = round(last_output_index * 0.66)
    # avoid using same output in very small models
    if secondary_output_index == last_output_index:
        secondary_output_index = secondary_output_index - 1

    model = Model(inputs=mo_model.inputs, outputs=[mo_model.outputs[secondary_output_index], mo_model.outputs[-1]])
    output_names = [k for i, k in enumerate(mo_loss_weights.keys()) if i in [secondary_output_index, last_output_index]]
    loss_weights = {
        output_names[0]: 0.25,
        output_names[1]: 0.75
    }
    mo_losses = {k: v for k, v in mo_losses.items() if k in output_names}

    return model, mo_losses, loss_weights, output_names


def compile_post_search_model(mo_model: Model, model_gen: BaseModelGenerator, train_strategy: tf.distribute.Strategy,
                              enable_xla: bool, extra_metrics: 'Optional[list[metrics.Metric]]' = None):
    '''
    Build a model suited for final evaluation, given a multi-output model and the model generator with the correct macro parameters.
    '''
    mo_losses, mo_loss_weights, optimizer, train_metrics = model_gen.define_training_hyperparams_and_metrics()
    # remove unnecessary exits and recalibrate loss weights
    model, mo_losses, loss_weights, output_names = prune_excessive_outputs(mo_model, mo_losses, mo_loss_weights)

    if extra_metrics is not None:
        train_metrics.extend(extra_metrics)

    execution_steps = get_optimized_steps_per_execution(train_strategy)
    model.compile(optimizer=optimizer, loss=mo_losses, loss_weights=loss_weights, metrics=train_metrics,
                  steps_per_execution=execution_steps, jit_compile=enable_xla)
    return model, output_names


def build_macro_customized_config(config: RunConfig, macro: MacroConfig):
    model_config = copy.deepcopy(config)
    model_config.architecture_hyperparameters.motifs = macro.m
    model_config.architecture_hyperparameters.normal_cells_per_motif = macro.n
    model_config.training_hyperparameters.filters = macro.f

    return model_config


def save_evaluation_results(model: Model, ds: tf.data.Dataset, model_path: str):
    results = model.evaluate(x=ds, return_dict=True)
    with open(os.path.join(model_path, 'eval.txt'), 'w') as f:
        f.write(f'Results: {results}')


def get_best_cell_specs(log_folder_path: str, n: int, metric: TargetMetric) -> 'tuple[list[CellSpecification], list[float]]':
    training_results_csv_path = os.path.join(log_folder_path, 'csv', 'training_results.csv')
    df = pd.read_csv(training_results_csv_path)
    # exclude empty cell, which gives exceptions in macro-structure tuning
    # could happen to choose the empty cell in debug runs with about 10 networks; in real cases it never happens
    df = df[df['cell structure'] != '[]']
    best_acc_rows = df.nlargest(n, columns=[metric.results_csv_column])

    cell_specs = parse_cell_strings(best_acc_rows['cell structure'])
    return cell_specs, best_acc_rows[metric.results_csv_column].to_list()
