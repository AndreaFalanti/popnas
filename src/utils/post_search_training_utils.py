'''
Stores common function used in training scripts, which could be executed after the main search procedure.
All the functions inside this file are not used inside the search algorithm.
'''
import copy
import json
import logging
import operator
import os
from typing import Optional, NamedTuple, Any

import tensorflow as tf
from tensorflow.keras import callbacks, Model, metrics, losses

import log_service
from models.model_generator import ModelGenerator
from utils.func_utils import create_empty_folder
from utils.nn_utils import get_multi_output_best_epoch_stats, initialize_train_strategy, get_optimized_steps_per_execution
from utils.rstr import rstr


def define_callbacks(score_metric: str, multi_output: bool, last_cell_index: int) -> 'list[callbacks.Callback]':
    '''
    Define callbacks used in model training.

    Returns:
        (tf.keras.Callback[]): Keras callbacks
    '''
    # Save best weights
    target_metric = f'val_Softmax_c{last_cell_index}_{score_metric}' if multi_output else f'val_{score_metric}'
    ckpt_save_format = 'cp_e{epoch:02d}_vl{val_loss:.2f}_v' + score_metric[:2] + '{' + target_metric + ':.4f}.ckpt'
    ckpt_callback = callbacks.ModelCheckpoint(filepath=log_service.build_path('weights', ckpt_save_format),
                                              save_weights_only=True, save_best_only=True, monitor=target_metric, mode='max')
    # By default, it shows losses and metrics for both training and validation
    tb_callback = callbacks.TensorBoard(log_dir=log_service.build_path('tensorboard'), profile_batch=0, histogram_freq=0)

    return [ckpt_callback, tb_callback]


def log_best_cell_results_during_search(logger: logging.Logger, cell_spec: list, best_score: Optional[float], metric: str, index: int = 0):
    logger.info('%s', f' CELL INFO ({index}) ')
    logger.info('Cell specification:')
    for i, block in enumerate(cell_spec):
        logger.info("\tBlock %d: %s", i + 1, rstr(block))
    if best_score is not None:
        logger.info('Best score (%s) reached during training: %0.4f', metric, best_score)


def create_model_log_folder(log_path: str):
    create_empty_folder(log_path)
    os.mkdir(os.path.join(log_path, 'weights'))  # create weights folder
    os.mkdir(os.path.join(log_path, 'tensorboard'))  # create tensorboard folder

    log_service.set_log_path(log_path)


def save_trimmed_json_config(config: dict, save_path: str):
    # remove useless keys (config is a subset of search algorithm config)
    keep_keys = ['cnn_hp', 'architecture_parameters', 'dataset', 'search_strategy']
    deletable_keys = [k for k in config.keys() if k not in keep_keys]

    for key in deletable_keys:
        del config[key]

    with open(os.path.join(save_path, 'run.json'), 'w') as f:
        json.dump(config, f, indent=4)


def log_final_training_results(logger: logging.Logger, hist: callbacks.History, score_metric: str, training_time: float, using_multi_output: bool,
                               using_val: bool = True):
    # hist.history is a dictionary of lists (each metric is a key)
    if using_multi_output:
        epoch_index, best_val_score, epoch_metrics_per_output = get_multi_output_best_epoch_stats(hist, score_metric, using_val)
    else:
        hist_metric = f'val_{score_metric}' if using_val else score_metric
        epoch_index, best_val_score = max(enumerate(hist.history[hist_metric]), key=operator.itemgetter(1))
        epoch_metrics_per_output = {'best': {}}
        if using_val:
            epoch_metrics_per_output['best']['val_loss'] = hist.history[f'val_loss'][epoch_index]
            epoch_metrics_per_output['best']['val_acc'] = hist.history[f'val_accuracy'][epoch_index]
            epoch_metrics_per_output['best']['val_top_k'] = hist.history[f'val_top_k_categorical_accuracy'][epoch_index]
            epoch_metrics_per_output['best']['val_f1'] = hist.history[f'val_f1_score'][epoch_index]

        epoch_metrics_per_output['best']['loss'] = hist.history[f'loss'][epoch_index]
        epoch_metrics_per_output['best']['acc'] = hist.history[f'accuracy'][epoch_index]
        epoch_metrics_per_output['best']['top_k'] = hist.history[f'top_k_categorical_accuracy'][epoch_index]
        epoch_metrics_per_output['best']['f1'] = hist.history[f'f1_score'][epoch_index]

    logger.info('*' * 31 + ' TRAINING SUMMARY ' + '*' * 31)
    logger.info('Best epoch index: %d', epoch_index + 1)
    logger.info('Total epochs: %d', len(hist.epoch))  # should work also for early stopping
    logger.info('Best validation %s: %0.4f', score_metric, best_val_score)
    logger.info('Total training time (without callbacks): %0.4f seconds (%d hours %d minutes %d seconds)',
                training_time, training_time // 3600, (training_time // 60) % 60, training_time % 60)

    for key in epoch_metrics_per_output:
        log_title = ' BEST EPOCH ' if key == 'best' else f' Cell {key} '
        logger.info('*' * 36 + log_title + '*' * 36)

        if using_val:
            logger.info('Validation')
            logger.info('\tValidation accuracy: %0.4f', epoch_metrics_per_output[key]['val_acc'])
            logger.info('\tValidation loss: %0.4f', epoch_metrics_per_output[key]['val_loss'])
            logger.info('\tValidation top_k accuracy: %0.4f', epoch_metrics_per_output[key]['val_top_k'])
            logger.info('\tValidation average f1 score: %0.4f', epoch_metrics_per_output[key]['val_f1'])

        logger.info('Training')
        logger.info('\tAccuracy: %0.4f', epoch_metrics_per_output[key]['acc'])
        logger.info('\tLoss: %0.4f', epoch_metrics_per_output[key]['loss'])
        logger.info('\tTop_k accuracy: %0.4f', epoch_metrics_per_output[key]['top_k'])
        logger.info('\tAverage f1 score: %0.4f', epoch_metrics_per_output[key]['f1'])

    logger.info('*' * 80)
    return best_val_score


def build_config(args, custom_json_path: str):
    # run configuration used during search. Dataset config is extracted from this.
    with open(os.path.join(args.p, 'restore', 'run.json'), 'r') as f:
        search_config = json.load(f)  # type: dict
    # read hyperparameters to use for model selection
    with open(custom_json_path, 'r') as f:
        config = json.load(f)  # type: dict

    def merge_config_section(section_name: str):
        ''' Override search config parameters with defined script config parameters, the ones not defined are taken from search config. '''
        return dict(search_config[section_name], **config[section_name])

    # override defined dataset parameters with script config, the ones not defined are taken from search config
    for key in config.keys():
        config[key] = merge_config_section(key)

    # set custom batch size, if present
    if args.b is not None:
        config['dataset']['batch_size'] = args.b
        config['cnn_hp']['learning_rate'] = search_config['cnn_hp']['learning_rate'] * (args.b / search_config['dataset']['batch_size'])

        # set score metric (to select best architecture if -spec is not provided)
    config['search_strategy'] = {
        'score_metric': search_config['search_strategy'].get('score_metric', 'accuracy')
    }

    # initialize train strategy (try-except to be retrocompatible with previous config format)
    try:
        ts_device = args.ts if args.ts is not None else search_config['others']['train_strategy']
        train_strategy = initialize_train_strategy(ts_device)
    except KeyError:
        train_strategy = initialize_train_strategy(None)

    return config, train_strategy


def prune_excessive_outputs(mo_model: Model, mo_loss_weights: 'dict[str, float]'):
    ''' Build a new model using only a secondary output at 2/3 of cells (drop other output from multi-output model) '''
    last_output_index = len(mo_model.outputs) - 1
    secondary_output_index = round(last_output_index * 0.66)
    model = Model(inputs=mo_model.inputs, outputs=[mo_model.outputs[secondary_output_index], mo_model.outputs[-1]])
    output_names = [k for i, k in enumerate(mo_loss_weights.keys()) if i in [secondary_output_index, last_output_index]]
    loss_weights = {
        output_names[0]: 0.25,
        output_names[1]: 0.75
    }

    return model, loss_weights


def override_checkpoint_callback(train_callbacks: list, score_metric: str, last_cell_index: int, use_val: bool = False):
    class ModelCheckpointCustom(callbacks.ModelCheckpoint):
        def on_epoch_end(self, epoch, logs=None):
            # at most 1 checkpoint every 10 epochs, the best one is saved (e.g. epoch 124 is best among [120, 129] -> saved in cp_ed12.ckpt
            super().on_epoch_end(epoch // 10, logs)

    # Save best weights, here we have no validation set, so we check the best on training
    prefix = 'val_' if use_val else ''
    target_metric = f'{prefix}Softmax_c{last_cell_index}_{score_metric}'

    ckpt_save_format = 'cp_ed{epoch:02d}.ckpt'
    train_callbacks[0] = ModelCheckpointCustom(filepath=log_service.build_path('weights', ckpt_save_format),
                                               save_weights_only=True, save_best_only=True, monitor=target_metric, mode='max')


class MacroConfig(NamedTuple):
    m: int
    n: int
    f: int

    def __str__(self) -> str:
        return f'm{self.m}-n{self.n}-f{self.f}'

    def modify(self, m_mod: int, n_mod: int, f_mod: float):
        return MacroConfig(self.m + m_mod, self.n + n_mod, int(self.f * f_mod))

    @staticmethod
    def from_config(config: 'dict[str, Any]'):
        return MacroConfig(config['architecture_parameters']['motifs'],
                           config['architecture_parameters']['normal_cells_per_motif'],
                           config['cnn_hp']['filters'])


def compile_post_search_model(mo_model: Model, model_gen: ModelGenerator, train_strategy: tf.distribute.Strategy):
    '''
    Build a model suited for final evaluation, given a multi-output model and the model generator with the correct macro parameters.
    '''
    loss, mo_loss_weights, optimizer, train_metrics = model_gen.define_training_hyperparams_and_metrics()
    train_metrics.append(metrics.TopKCategoricalAccuracy(k=5))

    # enable label smoothing
    loss = losses.CategoricalCrossentropy(label_smoothing=0.1)

    # remove unnecessary exits and recalibrate loss weights
    model, loss_weights = prune_excessive_outputs(mo_model, mo_loss_weights)

    execution_steps = get_optimized_steps_per_execution(train_strategy)
    model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights, metrics=train_metrics, steps_per_execution=execution_steps)
    return model


def build_macro_customized_config(config: dict, macro: MacroConfig):
    model_config = copy.deepcopy(config)
    model_config['architecture_parameters']['motifs'] = macro.m
    model_config['architecture_parameters']['normal_cells_per_motif'] = macro.n
    model_config['cnn_hp']['filters'] = macro.f

    return model_config
