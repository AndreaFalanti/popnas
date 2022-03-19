import argparse
import json
import logging
import os
from pathlib import Path

import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import metrics

import log_service
from model import ModelGenerator
from utils.dataset_utils import generate_tensorflow_datasets
from utils.func_utils import parse_cell_structures
from utils.rstr import rstr

# disable Tensorflow info and warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # disable Tensorflow info messages
tf.get_logger().setLevel(logging.ERROR)

AUTOTUNE = tf.data.AUTOTUNE


def get_best_cell_spec(log_folder_path: str):
    training_results_csv_path = os.path.join(log_folder_path, 'csv', 'training_results.csv')
    df = pd.read_csv(training_results_csv_path)
    best_acc_row = df.loc[df['best val accuracy'].idxmax()]

    cell_spec = parse_cell_structures([best_acc_row['cell structure']])[0]
    return cell_spec, best_acc_row['best val accuracy']


def main():
    # spec argument can be taken from model summary.txt, changing commas between tuples with ;
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-p', metavar='PATH', type=str, help="path to log folder", required=True)
    parser.add_argument('-j', metavar='JSON_PATH', type=str, help='path to config json with training parameters', default=None)
    parser.add_argument('-spec', metavar='CELL_SPECIFICATION', type=str, help="cell specification string", default=None)
    parser.add_argument('--stem', help='add ImageNet stem to network architecture', action='store_true')
    args = parser.parse_args()

    model_path = os.path.join(args.p, 'best_model_training')
    log_service.set_log_path(model_path)
    logger = log_service.get_logger(__name__, 'eval.log')

    custom_json_path = Path(__file__).parent / '../configs/final_training.json' if args.j is None else args.j
    logger.info('Reading configuration...')
    with open(custom_json_path, 'r') as f:
        config = json.load(f)

    cnn_config = config['cnn_hp']
    arc_config = config['architecture_parameters']

    # Load and prepare the dataset
    logger.info('Preparing datasets...')
    dataset_folds, classes_count, image_shape, train_batches, val_batches = generate_tensorflow_datasets(config['dataset'], logger)
    logger.info('Datasets generated successfully')

    # Generate the model
    cell_spec, _ = get_best_cell_spec(args.p) if args.spec is None else parse_cell_structures([args.spec])[0]
    logger.info('Cell specification:')
    for i, block in enumerate(cell_spec):
        logger.info("Block %d: %s", i + 1, rstr(block))

    logger.info('Generating Keras model from cell specification...')

    model_gen = ModelGenerator(cnn_config, arc_config, train_batches, output_classes=classes_count, image_shape=image_shape,
                               data_augmentation_model=None)
    model, _, last_cell_index = model_gen.build_model(cell_spec, add_imagenet_stem=args.stem)

    loss, loss_weights, optimizer, train_metrics = model_gen.define_training_hyperparams_and_metrics()
    train_metrics.append(metrics.TopKCategoricalAccuracy(k=5))
    train_metrics.append(tfa.metrics.F1Score(num_classes=classes_count))

    model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights, metrics=train_metrics)

    logger.info('Model generated successfully')

    # model.summary(line_length=140, print_fn=logger.info)

    train_dataset, validation_dataset = dataset_folds[0]

    latest = tf.train.latest_checkpoint(os.path.join(model_path, 'weights'))
    model.load_weights(latest)
    logger.info('Weights loaded successfully')

    results = model.evaluate(x=validation_dataset, return_dict=True)

    logger.info('Results: %s', rstr(results))


if __name__ == '__main__':
    main()
