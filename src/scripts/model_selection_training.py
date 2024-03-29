import argparse
import csv
import logging
import os
from pathlib import Path
from typing import Iterator

import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras.utils import plot_model

import log_service
from dataset.generators.factory import dataset_generator_factory
from dataset.utils import generate_balanced_weights_for_classes
from models.custom_callbacks.training_time import TrainingTimeCallback
from models.generators import BaseModelGenerator
from models.generators.factory import model_generator_factory
from models.results.base import TargetMetric
from search_space_units import CellSpecification
from utils.func_utils import create_empty_folder
from utils.nn_utils import save_keras_model_to_onnx, predict_and_save_confusion_matrix, perform_global_memory_clear, \
    remove_annoying_tensorflow_messages
from utils.post_search_training_utils import create_model_log_folder, dump_json_config, define_callbacks, \
    build_config, MacroConfig, compile_post_search_model, build_macro_customized_config, \
    get_best_cell_specs, extend_keras_metrics, extract_final_training_results, log_training_results_summary, \
    log_training_results_dict

# disable Tensorflow info and warning messages
remove_annoying_tensorflow_messages()


def get_cells_to_train_iter(run_path: str, spec: str, top_cells: int, score_metric: TargetMetric, macro: MacroConfig,
                            logger: logging.Logger) -> Iterator['tuple[int, tuple[CellSpecification, MacroConfig]]']:
    '''
    Return an iterator of all cells to train during the model selection process.

    Args:
        run_path: path to run folder
        spec: cell specification, if provided
        top_cells: number of top cells to consider in post-search training
        score_metric: TargetMetric for selecting top-k architectures
        macro: macro parameters used during search
        logger:

    Returns:
        Enumerate generator, in form: (cell_index, (cell_spec, score_during_search, macro_config))
    '''
    if spec is None:
        logger.info('Getting best cell specifications found during POPNAS run...')

        cell_specs, best_scores = get_best_cell_specs(run_path, top_cells, score_metric)
        logger.info('Top %d cells found:', top_cells)
        for i, (cell_spec, score) in enumerate(zip(cell_specs, best_scores)):
            logger.info('\t(%d) %s: %0.4f, cell spec: %s', i, score_metric.name, score, str(cell_spec))

        return enumerate(zip(cell_specs, [macro] * len(cell_specs))).__iter__()
    # single cell specification given by the user
    else:
        return enumerate(zip([CellSpecification.from_str(spec)], [macro])).__iter__()


def add_macro_architecture_changes_to_cells_iter(cells_iter: Iterator['tuple[int, tuple[CellSpecification, MacroConfig]]'],
                                                 original_macro: MacroConfig, min_params: int, max_params: int,
                                                 model_gen: BaseModelGenerator) -> Iterator['tuple[int, tuple[CellSpecification, MacroConfig]]']:
    m_modifiers = [0, 1]
    n_modifiers = [0, 1, 2]
    f_modifiers = [0.85, 1, 1.5, 1.75, 2]

    for i, (cell_spec, macro) in cells_iter:
        # always generate the original architecture
        yield i, (cell_spec, macro)

        # return the max number of filters for each (M, N) combination that fits parameter constraints
        for m_mod in m_modifiers:
            for n_mod in n_modifiers:
                max_f_macro = None
                for f_mod in f_modifiers:
                    new_macro = original_macro.modify(m_mod, n_mod, f_mod)
                    model_gen.alter_macro_structure(*new_macro)

                    # f_mods are ordered, so the last one to satisfy the condition is the max f
                    if min_params < model_gen.build_model_graph(cell_spec).get_total_params() < max_params:
                        max_f_macro = new_macro

                # check also that it is different from the original macro
                if max_f_macro is not None and str(max_f_macro) != str(macro):
                    yield i, (cell_spec, max_f_macro)


def execute(p: str, j: str = None, k: int = 5, spec: str = None, b: int = None, params: str = None, ts: str = None,
            name: str = 'best_model_training', debug: bool = False, stem: bool = False):
    ''' Refer to argparse help for more information about these arguments. '''
    # create appropriate model structure (different between single model and multi-models)
    if k <= 1:
        save_path = os.path.join(p, name)
        create_model_log_folder(save_path)
    else:
        save_path = os.path.join(p, f'{name}_top{k}')
        # avoid generating tensorboard and weights folders
        create_empty_folder(save_path)
        log_service.set_log_path(save_path)

    logger = log_service.get_logger(__name__)
    # reconnect to the Neptune project instantiated during the search procedure (identified by name of experiment folder)
    log_service.restore_neptune_project(run_name=os.path.split(p)[1])

    # NOTE: it's bugged on windows, see https://github.com/tensorflow/tensorflow/issues/43608. Run debug only on linux.
    if debug:
        tf.debugging.experimental.enable_dump_debug_info(os.path.join(save_path, 'debug'),
                                                         tensor_debug_mode="FULL_HEALTH",
                                                         circular_buffer_size=-1)

    custom_json_path = Path(__file__).parent / '../configs/model_selection_training.json' if j is None else j
    if log_service.neptune_project is not None:
        log_service.neptune_project['model_selection_json_override'].upload(custom_json_path)
        log_service.neptune_project['params_range'] = str(params)
        log_service.neptune_project['num_models_considered'] = k

    logger.info('Reading configuration...')
    config, train_strategy = build_config(p, b, ts, custom_json_path)

    train_config = config.training_hyperparameters
    arc_config = config.architecture_hyperparameters
    ds_config = config.dataset

    multi_output = arc_config.multi_output
    score_metric_name = config.search_strategy.score_metric

    # dump the json into save folder, so that is possible to retrieve how the model had been trained
    dump_json_config(config, save_path)

    # Load and prepare the dataset
    logger.info('Preparing datasets...')
    dataset_generator = dataset_generator_factory(ds_config, config.others)
    dataset_folds, classes_count, input_shape, train_batches, val_batches, preprocessing_model = dataset_generator.generate_train_val_datasets()

    # produce weights for balanced loss if option is enabled in database config
    balanced_class_weights = [generate_balanced_weights_for_classes(train_ds) for train_ds, _ in dataset_folds] \
        if ds_config.balance_class_losses else [None] * len(dataset_folds)
    logger.info('Datasets generated successfully')

    # DEBUG ONLY
    # test_data_augmentation(dataset_folds[0][0])

    # create a model generator instance
    with train_strategy.scope():
        model_gen = model_generator_factory(ds_config, train_config, arc_config, train_batches,
                                            output_classes_count=classes_count, input_shape=input_shape,
                                            preprocessing_model=preprocessing_model)

    keras_metrics = model_gen.get_results_processor_class().keras_metrics_considered()
    extended_keras_metrics = extend_keras_metrics(keras_metrics)
    target_metric = next(m for m in keras_metrics if m.name == score_metric_name)

    m = arc_config.motifs
    n = arc_config.normal_cells_per_motif
    f = arc_config.filters
    macro_config = MacroConfig(m, n, f)

    cell_score_iter = get_cells_to_train_iter(p, spec, k, target_metric, macro_config, logger)

    # generate alternative macro architectures of the selected cell specifications.
    # the returned iterator as the same structure and returns also the original elements
    if params is not None:
        # get params ranges. float conversion is used to support exponential notation (e.g. 2e6 for 2 millions).
        min_params, max_params = map(int, map(float, params.split(';')))

        cell_score_iter = add_macro_architecture_changes_to_cells_iter(cell_score_iter, macro_config, min_params, max_params, model_gen)

    for i, (cell_spec, macro) in cell_score_iter:
        model_folder = os.path.join(save_path, f'{i}-{macro}')
        create_model_log_folder(model_folder)
        model_id = f'model_{i}_{macro}'
        model_logger = log_service.get_logger(model_id)

        model_logger.info('CELL INFO (%d)', i)
        cell_spec.pretty_logging(logger)

        logger.info('Executing model %d-%s training', i, macro)
        # write cell spec to external file, stored together with results (usable by other scripts and to remember what cell has been trained)
        with open(os.path.join(model_folder, 'cell_spec.txt'), 'w') as f:
            f.write(str(cell_spec))

        model_config = build_macro_customized_config(config, macro)
        dump_json_config(model_config, model_folder)

        model_logger.info('Generating Keras model from cell specification...')
        with train_strategy.scope():
            # alter macro parameters of the model generator, before building the model
            model_gen.alter_macro_structure(*macro)

            mo_model, output_names = model_gen.build_model(cell_spec, add_imagenet_stem=stem)
            model, output_names = compile_post_search_model(mo_model, model_gen, train_strategy,
                                                            enable_xla=model_config.others.enable_XLA_compilation)

        model_logger.info('Model generated successfully')
        model.summary(line_length=140, print_fn=model_logger.info)

        model_logger.info('Converting untrained model to ONNX')
        save_keras_model_to_onnx(model, save_path=os.path.join(model_folder, 'untrained.onnx'))

        # TODO: right now only the first fold is used, expand the logic later to support multiple folds
        train_dataset, validation_dataset = dataset_folds[0]

        train_callbacks = define_callbacks(score_metric_name, output_names, use_val=True)
        neptune_run, train_callbacks = log_service.generate_neptune_run(f'ms:{model_id}', ['model-selection'], cell_spec, train_callbacks)
        log_service.save_macro_hp_in_neptune_run(neptune_run, macro.m, macro.n, macro.f)

        time_cb = TrainingTimeCallback()
        train_callbacks.insert(0, time_cb)

        plot_model(model, to_file=os.path.join(model_folder, 'model.pdf'), show_shapes=True, show_layer_names=True)

        hist = model.fit(x=train_dataset,
                         epochs=train_config.epochs,
                         steps_per_epoch=train_batches,
                         validation_data=validation_dataset,
                         validation_steps=val_batches,
                         class_weight=balanced_class_weights[0],
                         callbacks=train_callbacks)  # type: callbacks.History

        training_time = time_cb.get_total_time()
        results_dict, best_epoch, best_training_score = extract_final_training_results(hist, score_metric_name, extended_keras_metrics,
                                                                                       output_names, using_val=True)
        log_training_results_summary(logger, best_epoch, train_config.epochs, training_time, best_training_score, score_metric_name)
        log_training_results_dict(logger, results_dict)
        log_service.finalize_neptune_run(neptune_run, model.count_params(), training_time)

        model_logger.info('Converting trained model to ONNX')
        save_keras_model_to_onnx(model, save_path=os.path.join(model_folder, 'trained.onnx'))

        # create confusion matrix only in classification tasks
        if ds_config.type in ['image_classification', 'time_series_classification']:
            logger.info('Saving confusion matrix')
            predict_and_save_confusion_matrix(model, validation_dataset, multi_output, n_classes=classes_count,
                                              save_path=os.path.join(model_folder, 'val_confusion_matrix'))

        logger.info('Model %d-%s training complete', i, macro)
        perform_global_memory_clear()

        # keep track of all results
        with open(os.path.join(save_path, 'training_results.csv'), 'a', newline='') as f:
            writer = csv.writer(f)
            # append mode, so if file handler is in position 0 it means is empty. In this case, write the headers too.
            if f.tell() == 0:
                writer.writerow(['cell_spec', 'm', 'n', 'f', 'best_epoch', 'val_score', 'training_time'])

            writer.writerow([str(cell_spec), *macro, best_epoch, best_training_score, training_time])


if __name__ == '__main__':
    # NOTE: spec argument can be taken from training_results.csv
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-p', metavar='PATH', type=str, help="path to log folder", required=True)
    parser.add_argument('-j', metavar='JSON_PATH', type=str, help='path to config json with training parameters', default=None)
    parser.add_argument('-k', metavar='NUM_MODELS', type=int, help='number of top models to train, when -spec is not specified', default=5)
    parser.add_argument('-b', metavar='BATCH SIZE', type=int, help="desired batch size", default=None)
    parser.add_argument('-params', metavar='PARAMS RANGE', type=str, help="desired params range, semicolon separated (e.g. 2.5e6;3.5e6)",
                        default=None)
    parser.add_argument('-ts', metavar='TRAIN_STRATEGY', type=str, help='device used in Tensorflow distribute strategy', default=None)
    parser.add_argument('-spec', metavar='CELL_SPECIFICATION', type=str, help="cell specification string", default=None)
    parser.add_argument('-name', metavar='OUTPUT_NAME', type=str, help="output location in log folder", default='best_model_training')
    parser.add_argument('--debug', help='produce debug files of the whole training procedure', action='store_true')
    parser.add_argument('--stem', help='add ImageNet stem to network architecture', action='store_true')
    args = parser.parse_args()

    execute(**vars(args))
