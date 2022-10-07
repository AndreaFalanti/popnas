import argparse
import json
import logging
import os
from pathlib import Path

import pandas as pd
import tensorflow as tf
from tensorflow.keras import callbacks, models, metrics
from tensorflow.python.keras.utils.vis_utils import plot_model

import log_service
from dataset.augmentation import get_image_data_augmentation_model
from dataset.utils import dataset_generator_factory, generate_balanced_weights_for_classes, test_data_augmentation
from models.model_generator import ModelGenerator
from utils.feature_utils import metrics_fields_dict
from utils.final_training_utils import create_model_log_folder, save_trimmed_json_config, log_best_cell_results_during_search, define_callbacks, \
    log_final_training_results
from utils.func_utils import parse_cell_structures, cell_spec_to_str
from utils.nn_utils import initialize_train_strategy, get_optimized_steps_per_execution, save_keras_model_to_onnx
from utils.timing_callback import TimingCallback

# disable Tensorflow info and warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # disable Tensorflow info messages
tf.get_logger().setLevel(logging.ERROR)

AUTOTUNE = tf.data.AUTOTUNE


def get_best_cell_spec(log_folder_path: str, metric: str = 'best val accuracy'):
    training_results_csv_path = os.path.join(log_folder_path, 'csv', 'training_results.csv')
    df = pd.read_csv(training_results_csv_path)
    best_acc_row = df.loc[df[metric].idxmax()]

    cell_spec = parse_cell_structures([best_acc_row['cell structure']])[0]
    return cell_spec, best_acc_row[metric]


def main():
    # spec argument can be taken from model summary.txt, changing commas between tuples with ;
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-p', metavar='PATH', type=str, help="path to log folder", required=True)
    parser.add_argument('-j', metavar='JSON_PATH', type=str, help='path to config json with training parameters', default=None)
    parser.add_argument('-ts', metavar='TRAIN_STRATEGY', type=str, help='device used in Tensorflow distribute strategy', default=None)
    parser.add_argument('-spec', metavar='CELL_SPECIFICATION', type=str, help="cell specification string", default=None)
    parser.add_argument('-name', metavar='OUTPUT_NAME', type=str, help="output location in log folder", default='best_model_training')
    parser.add_argument('--load', help='load model from checkpoint', action='store_true')
    parser.add_argument('--same', help='use same hyperparams of the ones used during search algorithm', action='store_true')
    parser.add_argument('--debug', help='produce debug files of the whole training procedure', action='store_true')
    parser.add_argument('--stem', help='add ImageNet stem to network architecture', action='store_true')
    args = parser.parse_args()

    if args.same and args.j is not None:
        raise AttributeError("Can't specify both 'j' and 'same' arguments, they are mutually exclusive")

    custom_json_path = Path(__file__).parent / '../configs/model_selection_training.json' if args.j is None else args.j

    save_path = os.path.join(args.p, args.name)
    create_model_log_folder(save_path)
    logger = log_service.get_logger(__name__)

    # NOTE: it's bugged on windows, see https://github.com/tensorflow/tensorflow/issues/43608. Run debug only on linux.
    if args.debug:
        tf.debugging.experimental.enable_dump_debug_info(
            os.path.join(save_path, 'debug'),
            tensor_debug_mode="FULL_HEALTH",
            circular_buffer_size=-1)

    with open(os.path.join(args.p, 'restore', 'run.json'), 'r') as f:
        run_config = json.load(f)  # type: dict

    logger.info('Reading configuration...')
    if args.same:
        config = run_config
    else:
        with open(custom_json_path, 'r') as f:
            config = json.load(f)   # type: dict

    # initialize train strategy
    # retrocompatible with previous config format, which have no "others" section
    config_ts_device = config['others'].get('train_strategy', None) if 'others' in config.keys() else config.get('train_strategy', None)
    ts_device = args.ts if args.ts is not None else config_ts_device
    train_strategy = initialize_train_strategy(ts_device)

    cnn_config = config['cnn_hp']
    arc_config = config['architecture_parameters']

    cdr_enabled = cnn_config['cosine_decay_restart']['enabled']
    multi_output = arc_config['multi_output']
    augment_on_gpu = config['dataset']['data_augmentation']['perform_on_gpu']
    # expand number of epochs when training with same settings of the search algorithm, otherwise we would perform the same training
    # with these setting we have 7 periods of cosine decay restart (initial period = 2 epochs)
    epochs = (254 if cdr_enabled else 300) if args.same else cnn_config['epochs']
    cnn_config['cosine_decay_restart']['period_in_epochs'] = 2

    # enable cutout
    config['dataset']['data_augmentation']['use_cutout'] = True

    # dump the json into save folder, so that is possible to retrieve how the model had been trained
    # update and prune JSON config first (especially when coming from --same flag since it has all params of search algorithm)
    cnn_config['epochs'] = epochs
    save_trimmed_json_config(config, save_path)
    score_metric = config['search_strategy'].get('score_metric', 'accuracy')

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

    # TODO: load model from checkpoint is more of a legacy feature right now. Delete it?
    if args.load:
        logger.info('Loading best model from provided folder...')
        with train_strategy.scope():
            model = models.load_model(os.path.join(args.p, 'best_model', 'tf_model'))  # type: models.Model

        last_cell_index = 7     # TODO
        logger.info('Model loaded successfully')
    else:
        if args.spec is None:
            logger.info('Getting best cell specification found during POPNAS run...')
            # find best model found during search and log some relevant info
            metric = metrics_fields_dict[score_metric].real_column
            cell_spec, best_score = get_best_cell_spec(args.p, metric)
            log_best_cell_results_during_search(logger, cell_spec, best_score, metric)
        else:
            cell_spec = parse_cell_structures([args.spec])[0]

        logger.info('Generating Keras model from cell specification...')
        # reconvert cell to str so that can be stored together with results (usable by other script and easier to remember what cell has been trained)
        with open(os.path.join(save_path, 'cell_spec.txt'), 'w') as f:
            f.write(cell_spec_to_str(cell_spec))

        with train_strategy.scope():
            model_gen = ModelGenerator(cnn_config, arc_config, train_batches, output_classes_count=classes_count, input_shape=input_shape,
                                       data_augmentation_model=get_image_data_augmentation_model() if augment_on_gpu else None)
            model, _, last_cell_index = model_gen.build_model(cell_spec, add_imagenet_stem=args.stem)

            loss, loss_weights, optimizer, train_metrics = model_gen.define_training_hyperparams_and_metrics()
            train_metrics.append(metrics.TopKCategoricalAccuracy(k=5))

            execution_steps = get_optimized_steps_per_execution(train_strategy)
            model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights, metrics=train_metrics, steps_per_execution=execution_steps)

        logger.info('Model generated successfully')

    model.summary(line_length=140, print_fn=logger.info)

    logger.info('Converting untrained model to ONNX')
    save_keras_model_to_onnx(model, save_path=os.path.join(save_path, 'untrained.onnx'))

    # TODO: right now only the first fold is used, expand the logic later to support multiple folds
    train_dataset, validation_dataset = dataset_folds[0]

    # Define callbacks
    train_callbacks = define_callbacks(cdr_enabled, score_metric, multi_output, last_cell_index)
    time_cb = TimingCallback()
    train_callbacks.append(time_cb)

    plot_model(model, to_file=os.path.join(save_path, 'model.pdf'), show_shapes=True, show_layer_names=True)

    hist = model.fit(x=train_dataset,
                     epochs=epochs,
                     steps_per_epoch=train_batches,
                     validation_data=validation_dataset,
                     validation_steps=val_batches,
                     class_weight=balanced_class_weights[0] if balance_class_losses else None,
                     callbacks=train_callbacks)     # type: callbacks.History

    training_time = sum(time_cb.logs)
    log_final_training_results(logger, hist, score_metric, training_time, arc_config['multi_output'])

    logger.info('Converting trained model to ONNX')
    save_keras_model_to_onnx(model, save_path=os.path.join(save_path, 'trained.onnx'))


if __name__ == '__main__':
    main()
