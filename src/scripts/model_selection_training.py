import argparse
import logging
import os
from pathlib import Path
from typing import Optional, Iterator

import pandas as pd
import tensorflow as tf
from tensorflow.keras import callbacks, metrics
from tensorflow.python.keras.utils.vis_utils import plot_model

import log_service
from dataset.augmentation import get_image_data_augmentation_model
from dataset.utils import dataset_generator_factory, generate_balanced_weights_for_classes
from models.model_generator import ModelGenerator
from utils.feature_utils import metrics_fields_dict
from utils.final_training_utils import create_model_log_folder, save_trimmed_json_config, log_best_cell_results_during_search, define_callbacks, \
    log_final_training_results, build_config
from utils.func_utils import parse_cell_structures, cell_spec_to_str
from utils.nn_utils import get_optimized_steps_per_execution, save_keras_model_to_onnx, predict_and_save_confusion_matrix
from utils.timing_callback import TimingCallback

# disable Tensorflow info and warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # disable Tensorflow info messages
tf.get_logger().setLevel(logging.ERROR)

AUTOTUNE = tf.data.AUTOTUNE


def get_best_cell_specs(log_folder_path: str, n: int, metric: str = 'best val accuracy'):
    training_results_csv_path = os.path.join(log_folder_path, 'csv', 'training_results.csv')
    df = pd.read_csv(training_results_csv_path)
    best_acc_rows = df.nlargest(n, columns=[metric])

    cell_specs = parse_cell_structures(best_acc_rows['cell structure'])
    return zip(cell_specs, best_acc_rows[metric])


def get_cells_to_train_iter(args, score_metric: str, logger: logging.Logger) -> Iterator['tuple[list, Optional[float]]']:
    if args.spec is None:
        logger.info('Getting best cell specification found during POPNAS run...')
        metric = metrics_fields_dict[score_metric].real_column

        return get_best_cell_specs(args.p, args.n, metric)
    # single cell specification given by the user
    else:
        return zip(parse_cell_structures([args.spec]), [None])


def main():
    # NOTE: spec argument can be taken from training_results.csv
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-p', metavar='PATH', type=str, help="path to log folder", required=True)
    parser.add_argument('-j', metavar='JSON_PATH', type=str, help='path to config json with training parameters', default=None)
    parser.add_argument('-n', metavar='NUM_MODELS', type=int, help='number of top models to train, when -spec is not specified', default=5)
    parser.add_argument('-b', metavar='BATCH SIZE', type=int, help="desired batch size", default=None)
    parser.add_argument('-ts', metavar='TRAIN_STRATEGY', type=str, help='device used in Tensorflow distribute strategy', default=None)
    parser.add_argument('-spec', metavar='CELL_SPECIFICATION', type=str, help="cell specification string", default=None)
    parser.add_argument('-name', metavar='OUTPUT_NAME', type=str, help="output location in log folder", default='best_model_training')
    parser.add_argument('--debug', help='produce debug files of the whole training procedure', action='store_true')
    parser.add_argument('--stem', help='add ImageNet stem to network architecture', action='store_true')
    args = parser.parse_args()

    custom_json_path = Path(__file__).parent / '../configs/model_selection_training.json' if args.j is None else args.j

    folder_name = args.name if args.spec is not None or args.n <= 1 else args.name + f'_top{args.n}'
    save_path = os.path.join(args.p, folder_name)
    create_model_log_folder(save_path)
    logger = log_service.get_logger(__name__)

    # NOTE: it's bugged on windows, see https://github.com/tensorflow/tensorflow/issues/43608. Run debug only on linux.
    if args.debug:
        tf.debugging.experimental.enable_dump_debug_info(
            os.path.join(save_path, 'debug'),
            tensor_debug_mode="FULL_HEALTH",
            circular_buffer_size=-1)

    logger.info('Reading configuration...')
    config, train_strategy = build_config(args, custom_json_path)

    cnn_config = config['cnn_hp']
    arc_config = config['architecture_parameters']

    multi_output = arc_config['multi_output']
    augment_on_gpu = config['dataset']['data_augmentation']['perform_on_gpu']
    score_metric = config['search_strategy']['score_metric']

    # dump the json into save folder, so that is possible to retrieve how the model had been trained
    save_trimmed_json_config(config, save_path)

    # Load and prepare the dataset
    logger.info('Preparing datasets...')
    dataset_generator = dataset_generator_factory(config['dataset'])
    dataset_folds, classes_count, input_shape, train_batches, val_batches = dataset_generator.generate_train_val_datasets()

    # produce weights for balanced loss if option is enabled in database config
    balance_class_losses = config['dataset'].get('balance_class_losses', False)
    balanced_class_weights = [generate_balanced_weights_for_classes(train_ds) for train_ds, _ in dataset_folds] \
        if balance_class_losses else None
    logger.info('Datasets generated successfully')

    # DEBUG ONLY
    # test_data_augmentation(dataset_folds[0][0])

    # create a model generator instance
    with train_strategy.scope():
        model_gen = ModelGenerator(cnn_config, arc_config, train_batches, output_classes_count=classes_count, input_shape=input_shape,
                                   data_augmentation_model=get_image_data_augmentation_model() if augment_on_gpu else None)

    cell_score_iter = get_cells_to_train_iter(args, score_metric, logger)

    for i, (cell_spec, best_score) in enumerate(cell_score_iter):
        model_folder = os.path.join(save_path, str(i))
        create_model_log_folder(model_folder)
        model_logger = log_service.get_logger(__name__)

        log_best_cell_results_during_search(model_logger, cell_spec, best_score, score_metric, i)

        logger.info('Executing model %d training', i)
        # reconvert cell to str so that can be stored together with results (usable by other script and easier to remember what cell has been trained)
        with open(os.path.join(model_folder, 'cell_spec.txt'), 'w') as f:
            f.write(cell_spec_to_str(cell_spec))

        save_trimmed_json_config(config, model_folder)

        model_logger.info('Generating Keras model from cell specification...')
        with train_strategy.scope():
            model, _, last_cell_index = model_gen.build_model(cell_spec, add_imagenet_stem=args.stem)

            loss, loss_weights, optimizer, train_metrics = model_gen.define_training_hyperparams_and_metrics()
            train_metrics.append(metrics.TopKCategoricalAccuracy(k=5))

            execution_steps = get_optimized_steps_per_execution(train_strategy)
            model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights, metrics=train_metrics, steps_per_execution=execution_steps)

        model_logger.info('Model generated successfully')

        model.summary(line_length=140, print_fn=model_logger.info)

        model_logger.info('Converting untrained model to ONNX')
        save_keras_model_to_onnx(model, save_path=os.path.join(model_folder, 'untrained.onnx'))

        # TODO: right now only the first fold is used, expand the logic later to support multiple folds
        train_dataset, validation_dataset = dataset_folds[0]

        # Define callbacks
        train_callbacks = define_callbacks(score_metric, multi_output, last_cell_index)
        time_cb = TimingCallback()
        train_callbacks.insert(0, time_cb)

        plot_model(model, to_file=os.path.join(model_folder, 'model.pdf'), show_shapes=True, show_layer_names=True)

        hist = model.fit(x=train_dataset,
                         epochs=cnn_config['epochs'],
                         steps_per_epoch=train_batches,
                         validation_data=validation_dataset,
                         validation_steps=val_batches,
                         class_weight=balanced_class_weights[0] if balance_class_losses else None,
                         callbacks=train_callbacks)  # type: callbacks.History

        training_time = sum(time_cb.logs)
        log_final_training_results(model_logger, hist, score_metric, training_time, arc_config['multi_output'])

        model_logger.info('Converting trained model to ONNX')
        save_keras_model_to_onnx(model, save_path=os.path.join(model_folder, 'trained.onnx'))

        logger.info('Saving confusion matrix')
        predict_and_save_confusion_matrix(model, validation_dataset, multi_output, n_classes=classes_count,
                                          save_path=os.path.join(model_folder, 'val_confusion_matrix'))

        logger.info('Model %d training complete', i)
        tf.keras.backend.clear_session()


if __name__ == '__main__':
    main()
