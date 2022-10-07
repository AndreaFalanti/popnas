import argparse
import json
import logging
import os
from pathlib import Path

import pandas as pd
import tensorflow as tf
from tensorflow.keras import callbacks, models, metrics, losses, Model
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


def override_checkpoint_callback(train_callbacks: list, score_metric: str, last_cell_index: int):
    class ModelCheckpointCustom(callbacks.ModelCheckpoint):
        def on_epoch_end(self, epoch, logs=None):
            # at most 1 checkpoint every 10 epochs, the best one is saved (e.g. epoch 124 is best among [120, 129] -> saved in cp_ed12.ckpt
            super().on_epoch_end(epoch // 10, logs)

    # Save best weights, here we have no validation set, so we check the best on training
    target_metric = f'Softmax_c{last_cell_index}_{score_metric}'

    ckpt_save_format = 'cp_ed{epoch:02d}.ckpt'
    train_callbacks[0] = ModelCheckpointCustom(filepath=log_service.build_path('weights', ckpt_save_format),
                                               save_weights_only=True, save_best_only=True, monitor=target_metric, mode='max')


def main():
    # spec argument can be taken from model summary.txt, changing commas between tuples with ;
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-p', metavar='PATH', type=str, help="path to log folder", required=True)
    parser.add_argument('-b', metavar='BATCH SIZE', type=int, help="desired batch size", required=True)
    parser.add_argument('-ts', metavar='TRAIN_STRATEGY', type=str, help='device used in Tensorflow distribute strategy', default=None)
    parser.add_argument('-spec', metavar='CELL_SPECIFICATION', type=str, help="cell specification string", default=None)
    parser.add_argument('-name', metavar='OUTPUT_NAME', type=str, help="output location in log folder", default='best_model_training')
    parser.add_argument('--stem', help='add ImageNet stem to network architecture', action='store_true')
    args = parser.parse_args()

    save_path = os.path.join(args.p, args.name)
    create_model_log_folder(save_path)
    logger = log_service.get_logger(__name__)

    logger.info('Reading configuration...')
    with open(os.path.join(args.p, 'restore', 'run.json'), 'r') as f:
        run_config = json.load(f)  # type: dict

    # read final_training config
    custom_json_path = Path(__file__).parent / '../configs/last_training.json'
    with open(custom_json_path, 'r') as f:
        config = json.load(f)  # type: dict

    # override part of the custom configuration with the search configuration
    config['dataset'] = run_config['dataset']
    config['dataset']['batch_size'] = args.b
    config['search_strategy'] = run_config['search_strategy']

    # enable cutout
    config['dataset']['data_augmentation']['use_cutout'] = True

    # initialize train strategy
    # retrocompatible with previous config format, which have no "others" section
    config_ts_device = config['others'].get('train_strategy', None) if 'others' in config.keys() else config.get('train_strategy', None)
    ts_device = args.ts if args.ts is not None else config_ts_device
    train_strategy = initialize_train_strategy(ts_device)

    cnn_config = config['cnn_hp']
    arc_config = config['architecture_parameters']

    cdr_enabled = False
    multi_output = True
    augment_on_gpu = False
    epochs = 600

    score_metric = config['search_strategy'].get('score_metric', 'accuracy')

    # Load and prepare the dataset
    logger.info('Preparing datasets...')
    dataset_generator = dataset_generator_factory(config['dataset'])
    train_ds, classes_count, input_shape, train_batches = dataset_generator.generate_final_training_dataset()

    # produce weights for balanced loss if option is enabled in database config
    balance_class_losses = config['dataset'].get('balance_class_losses', False)
    balanced_class_weights = generate_balanced_weights_for_classes(train_ds) if balance_class_losses else None
    logger.info('Datasets generated successfully')

    # DEBUG ONLY
    # test_data_augmentation(train_ds)

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
        mo_model, _, last_cell_index = model_gen.build_model(cell_spec, add_imagenet_stem=args.stem)

        loss, mo_loss_weights, optimizer, train_metrics = model_gen.define_training_hyperparams_and_metrics()
        train_metrics.append(metrics.TopKCategoricalAccuracy(k=5))

        # small changes to conform to PNAS training procedure
        # enable label smoothing
        loss = losses.CategoricalCrossentropy(label_smoothing=0.1)
        # build a new model using only a secondary output at 2/3 of cells (drop other output from multi-output model)
        secondary_output_index = int(last_cell_index * 0.66)
        model = Model(inputs=mo_model.inputs, outputs=[mo_model.outputs[secondary_output_index], mo_model.outputs[-1]])
        output_names = [k for i, k in enumerate(mo_loss_weights.keys()) if i in [secondary_output_index, last_cell_index]]
        loss_weights = {
            output_names[0]: 0.25,
            output_names[1]: 0.75
        }

        execution_steps = get_optimized_steps_per_execution(train_strategy)
        model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights, metrics=train_metrics, steps_per_execution=execution_steps)

    logger.info('Model generated successfully')

    model.summary(line_length=140, print_fn=logger.info)

    logger.info('Converting untrained model to ONNX')
    save_keras_model_to_onnx(model, save_path=os.path.join(save_path, 'untrained.onnx'))

    # Define callbacks
    train_callbacks = define_callbacks(cdr_enabled, score_metric, multi_output, last_cell_index)
    override_checkpoint_callback(train_callbacks, score_metric, last_cell_index)
    time_cb = TimingCallback()
    train_callbacks.insert(0, time_cb)

    plot_model(model, to_file=os.path.join(save_path, 'model.pdf'), show_shapes=True, show_layer_names=True)

    hist = model.fit(x=train_ds,
                     epochs=epochs,
                     steps_per_epoch=train_batches,
                     class_weight=balanced_class_weights,
                     callbacks=train_callbacks)     # type: callbacks.History

    training_time = sum(time_cb.logs)
    log_final_training_results(logger, hist, score_metric, training_time, arc_config['multi_output'], using_val=False)

    logger.info('Converting trained model to ONNX')
    save_keras_model_to_onnx(model, save_path=os.path.join(save_path, 'trained.onnx'))


if __name__ == '__main__':
    main()
