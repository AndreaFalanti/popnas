import argparse
import json
import logging
import os

import numpy as np
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras import metrics, models, Model

import log_service
from dataset.utils import dataset_generator_factory
from models.model_generator import ModelGenerator
from utils.func_utils import parse_cell_structures

# disable Tensorflow info and warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # disable Tensorflow info messages
tf.get_logger().setLevel(logging.ERROR)

AUTOTUNE = tf.data.AUTOTUNE


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: str, n_classes: int):
    cmat = confusion_matrix(y_true, y_pred)

    fig = plt.figure(figsize=(20, 20))
    if n_classes <= 20:
        sns.heatmap(cmat, annot=True, fmt='d')
    # avoid annotations in case there are too many classes
    else:
        sns.heatmap(cmat)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.savefig(save_path + '.png', bbox_inches='tight', dpi=120)
    plt.savefig(save_path + '.pdf', bbox_inches='tight')
    plt.close(fig)


def get_model_cell_spec(log_folder_path: str):
    with open(os.path.join(log_folder_path, 'cell_spec.txt'), 'r') as f:
        cell_spec = f.read()

    return parse_cell_structures([cell_spec])[0]


def evaluate_and_save_confusion_matrix(model: Model, test_ds: tf.data.Dataset, multi_output: bool, model_path: str, n_classes: int):
    results = model.evaluate(x=test_ds, return_dict=True)
    with open(os.path.join(model_path, 'eval.txt'), 'w') as f:
        f.write(f'Results: {results}')

    # plot confusion matrix. Y labels must be converted to integers and flatten (since they are batched)
    y_pred = model.predict(x=test_ds)
    # take last output, in case the model is multi-output
    if multi_output:
        y_pred = y_pred[-1]

    y_pred = np.argmax(y_pred, axis=-1).flatten()
    y_true = np.concatenate([np.argmax(y, axis=-1) for x, y in test_ds], axis=0)
    save_confusion_matrix(y_true, y_pred, save_path=os.path.join(model_path, 'confusion_matrix'), n_classes=n_classes)


# This script can be used to evaluate the final model trained on a test set.
# It needs a saved model, which could be the one found during search or the one produced by final_training script (spec + checkpoint)
def main():
    # if search_model flag is not specified, script will assume that the final_training script have been executed to train extensively a model
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-p', metavar='PATH', type=str, help="path to log folder", required=True)
    parser.add_argument('-j', metavar='JSON_PATH', type=str,
                        help='path to config json with training parameters (used to instantiate dataset)', default=None)
    parser.add_argument('--search_model', help='use best model found in search, with weights found on proxy training', action='store_true')
    parser.add_argument('-f', metavar='MODEL_FOLDER', type=str, help='model folder name (default: best_model_training)',
                        default='best_model_training')
    parser.add_argument('--top', help='when -f is provided, consider it a nested folder, output of top_k_final_training script', action='store_true')
    args = parser.parse_args()

    model_path = os.path.join(args.p, 'best_model') if args.search_model else os.path.join(args.p, args.f)
    log_service.set_log_path(model_path)

    if args.j:
        custom_json_path = args.j
    else:
        custom_json_path = os.path.join(args.p, 'restore', 'run.json') if args.search_model \
            else os.path.join(model_path, 'run.json')
    print('Reading configuration...')
    with open(custom_json_path, 'r') as f:
        config = json.load(f)

    cnn_config = config['cnn_hp']
    arc_config = config['architecture_parameters']
    multi_output = arc_config['multi_output']

    # Load and prepare the dataset
    print('Preparing datasets...')
    dataset_generator = dataset_generator_factory(config['dataset'])
    test_ds, classes_count, image_shape, test_batches = dataset_generator.generate_test_dataset()
    print('Datasets generated successfully')

    # Generate the model
    if args.search_model:
        model = models.load_model(os.path.join(args.p, 'best_model', 'tf_model'))
        print('Model loaded successfully from TF model files')
        evaluate_and_save_confusion_matrix(model, test_ds, multi_output, model_path=model_path, n_classes=classes_count)
    else:
        model_gen = ModelGenerator(cnn_config, arc_config, test_batches, output_classes_count=classes_count, input_shape=image_shape,
                                   data_augmentation_model=None)

        model_paths = [f.path for f in os.scandir(model_path) if f.is_dir()] if args.top else [model_path]
        for m_index, m_path in enumerate(model_paths):
            print(f'Processing model at "{m_path}"')

            cell_spec = get_model_cell_spec(m_path)
            print('Cell specification:')
            for i, block in enumerate(cell_spec):
                print(f'Block {i + 1}: {block}')

            print('Generating Keras model from cell specification...')

            model, _, last_cell_index = model_gen.build_model(cell_spec, add_imagenet_stem=False)

            loss, loss_weights, optimizer, train_metrics = model_gen.define_training_hyperparams_and_metrics()
            train_metrics.append(metrics.TopKCategoricalAccuracy(k=5))

            model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights, metrics=train_metrics)
            print('Model generated successfully')

            latest = tf.train.latest_checkpoint(os.path.join(m_path, 'weights'))
            model.load_weights(latest)
            print('Weights loaded successfully from checkpoint')

            evaluate_and_save_confusion_matrix(model, test_ds, multi_output, model_path=m_path, n_classes=classes_count)


if __name__ == '__main__':
    main()
