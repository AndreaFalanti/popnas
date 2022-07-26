import argparse
import json
import logging
import os

import tensorflow as tf
from tensorflow.keras import metrics, models

import log_service
from datasets.generator import generate_test_dataset
from model import ModelGenerator
from utils.func_utils import parse_cell_structures
from utils.rstr import rstr

# disable Tensorflow info and warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # disable Tensorflow info messages
tf.get_logger().setLevel(logging.ERROR)

AUTOTUNE = tf.data.AUTOTUNE


def get_model_cell_spec(log_folder_path: str):
    with open(os.path.join(log_folder_path, 'cell_spec.txt'), 'r') as f:
       cell_spec = f.read()

    return parse_cell_structures([cell_spec])[0]

# This script can be used to evaluate the final model trained on a test set.
# It needs a saved model, which could be the one found during search or the one produced by final_training script (spec + checkpoint)
def main():
    # if search_model flag is not specified, script will assume that the final_training script have been executed to train extensively a model
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-p', metavar='PATH', type=str, help="path to log folder", required=True)
    parser.add_argument('-j', metavar='JSON_PATH', type=str,
                        help='path to config json with training parameters (used to instantiate dataset)', default=None)
    parser.add_argument('--search_model', help='use best model found in search, with weights found on proxy training', action='store_true')
    parser.add_argument('-f', metavar='MODEL_FOLDER', type=str, help='model folder name (default: best_model_training)', default='best_model_training')
    args = parser.parse_args()

    model_path = os.path.join(args.p, 'best_model') if args.search_model else os.path.join(args.p, args.f)
    log_service.set_log_path(model_path)
    logger = log_service.get_logger(__name__, 'eval.log')

    if args.j:
        custom_json_path = args.j
    else:
        custom_json_path = os.path.join(args.p, 'restore', 'run.json') if args.search_model \
            else os.path.join(model_path, 'run.json')
    logger.info('Reading configuration...')
    with open(custom_json_path, 'r') as f:
        config = json.load(f)

    cnn_config = config['cnn_hp']
    arc_config = config['architecture_parameters']

    # Load and prepare the dataset
    logger.info('Preparing datasets...')
    test_ds, classes_count, image_shape, test_batches = generate_test_dataset(config['dataset'], logger)
    logger.info('Datasets generated successfully')

    # Generate the model
    if args.search_model:
        model = models.load_model(os.path.join(args.p, 'best_model', 'saved_model.h5'))
        logger.info('Model loaded successfully from H5 file')
    else:
        cell_spec = get_model_cell_spec(model_path)
        logger.info('Cell specification:')
        for i, block in enumerate(cell_spec):
            logger.info("Block %d: %s", i + 1, rstr(block))

        logger.info('Generating Keras model from cell specification...')

        model_gen = ModelGenerator(cnn_config, arc_config, test_batches, output_classes_count=classes_count, image_shape=image_shape,
                                   data_augmentation_model=None)
        model, _, last_cell_index = model_gen.build_model(cell_spec, add_imagenet_stem=False)

        loss, loss_weights, optimizer, train_metrics = model_gen.define_training_hyperparams_and_metrics()
        train_metrics.append(metrics.TopKCategoricalAccuracy(k=5))

        model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights, metrics=train_metrics)
        logger.info('Model generated successfully')

        # model.summary(line_length=140, print_fn=logger.info)

        latest = tf.train.latest_checkpoint(os.path.join(model_path, 'weights'))
        model.load_weights(latest)
        logger.info('Weights loaded successfully from checkpoint')

    results = model.evaluate(x=test_ds, return_dict=True)

    logger.info('Results: %s', rstr(results))


if __name__ == '__main__':
    main()
