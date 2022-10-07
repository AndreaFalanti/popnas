'''
Stores common function used in training scripts, which could be executed after the main search procedure.
All the functions inside this file are not used inside the search algorithm.
'''

import json
import logging
import operator
import os

from tensorflow.keras import callbacks

import log_service
from utils.func_utils import create_empty_folder
from utils.nn_utils import get_multi_output_best_epoch_stats
from utils.rstr import rstr


def define_callbacks(cdr_enabled: bool, score_metric: str, multi_output: bool, last_cell_index: int) -> 'list[callbacks.Callback]':
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
    # By default shows losses and metrics for both training and validation
    tb_callback = callbacks.TensorBoard(log_dir=log_service.build_path('tensorboard'), profile_batch=0, histogram_freq=0)

    # these callbacks are shared between all models
    train_callbacks = [ckpt_callback, tb_callback]

    # if using plain lr, adapt it with reduce learning rate on plateau
    # NOTE: for unknown reasons, activating plateau callback when cdr is present will also cause an error at the end of the first epoch
    # TODO: right now, if cdr is disabled the lr is scheduled with cosine decay, so reduceLROnPlateau could be not optimal.
    # if not cdr_enabled:
    #     train_callbacks.append(callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.4, patience=5, verbose=1, mode='max'))
    #     train_callbacks.append(callbacks.EarlyStopping(monitor='val_accuracy', patience=12, restore_best_weights=True, verbose=1, mode='max'))

    return train_callbacks


def log_best_cell_results_during_search(logger: logging.Logger, cell_spec: list, best_score: float, metric: str, index: int = 0):
    logger.info('%s', f' BEST CELL INFO ({index}) ')
    logger.info('Cell specification:')
    for i, block in enumerate(cell_spec):
        logger.info("\tBlock %d: %s", i + 1, rstr(block))
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
