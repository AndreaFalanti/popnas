import argparse
import json
import os

import tensorflow as tf
from tensorflow.keras import models

import log_service
from dataset.generators.factory import dataset_generator_factory
from models.generators.factory import model_generator_factory
from search_space import CellSpecification
from utils.config_utils import read_json_config
from utils.nn_utils import predict_and_save_confusion_matrix, initialize_train_strategy, perform_global_memory_clear, \
    remove_annoying_tensorflow_messages
from utils.post_search_training_utils import MacroConfig, compile_post_search_model, save_evaluation_results

# disable Tensorflow info and warning messages
remove_annoying_tensorflow_messages()

AUTOTUNE = tf.data.AUTOTUNE


def get_model_cell_spec(log_folder_path: str):
    with open(os.path.join(log_folder_path, 'cell_spec.txt'), 'r') as f:
        cell_spec = f.read()

    return CellSpecification.from_str(cell_spec)


# This script can be used to evaluate the final model trained on a test set.
# It needs a saved model, which could be the one found during search or the one produced by final_training script (spec + checkpoint)
def main():
    # if "search_model" flag is not specified, the script will assume that the final_training script has been executed to train extensively a model
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-p', metavar='PATH', type=str, help="path to log folder", required=True)
    parser.add_argument('-j', metavar='JSON_PATH', type=str,
                        help='path to config json with training parameters (used to instantiate dataset)', default=None)
    parser.add_argument('--search_model', help='use best model found in search, with weights found on proxy training', action='store_true')
    parser.add_argument('-f', metavar='MODEL_FOLDER', type=str, help='model folder name (default: best_model_training)',
                        default='best_model_training')
    parser.add_argument('-ts', metavar='TRAIN_STRATEGY', type=str, help='device used in Tensorflow distribute strategy', default=None)
    parser.add_argument('--top', help='when -f is provided, consider it as a nested folder if this option is set', action='store_true')
    args = parser.parse_args()

    model_path = os.path.join(args.p, 'best_model') if args.search_model else os.path.join(args.p, args.f)
    log_service.set_log_path(model_path)

    if args.j:
        custom_json_path = args.j
    else:
        custom_json_path = os.path.join(args.p, 'restore', 'run.json') if args.search_model \
            else os.path.join(model_path, 'run.json')
    print('Reading configuration...')
    config = read_json_config(custom_json_path)

    cnn_config = config.training_hyperparameters
    arc_config = config.architecture_hyperparameters
    multi_output = arc_config.multi_output

    train_strategy = initialize_train_strategy(args.ts)

    # Load and prepare the dataset
    print('Preparing datasets...')
    dataset_generator = dataset_generator_factory(config.dataset, config.others)
    test_ds, classes_count, image_shape, test_batches = dataset_generator.generate_test_dataset()
    print('Datasets generated successfully')

    # Generate the model
    if args.search_model:
        with train_strategy.scope():
            model = models.load_model(os.path.join(args.p, 'best_model', 'tf_model'))
        print('Model loaded successfully from TF model files')
        save_evaluation_results(model, test_ds, model_path)

        # create confusion matrix only in classification tasks
        if config.dataset.type in ['image_classification', 'time_series_classification']:
            predict_and_save_confusion_matrix(model, test_ds, multi_output, n_classes=classes_count,
                                              save_path=os.path.join(model_path, 'test_confusion_matrix'))
    else:
        with train_strategy.scope():
            model_gen = model_generator_factory(config.dataset, cnn_config, arc_config, test_batches,
                                                output_classes_count=classes_count, input_shape=image_shape, data_augmentation_model=None)

        model_paths = [f.path for f in os.scandir(model_path) if f.is_dir()] if args.top else [model_path]
        for m_index, m_path in enumerate(model_paths):
            print(f'Processing model at "{m_path}"')

            cell_spec = get_model_cell_spec(m_path)
            print('Cell specification:')
            for i, block in enumerate(cell_spec):
                print(f'Block {i + 1}: {block}')

            print('Generating Keras model from cell specification...')

            # read model configuration to extract its macro architecture parameters
            with open(os.path.join(m_path, 'run.json'), 'r') as f:
                model_config = json.load(f)

            macro = MacroConfig.from_config(model_config)

            with train_strategy.scope():
                model_gen.alter_macro_structure(*macro)
                mo_model, _ = model_gen.build_model(cell_spec, add_imagenet_stem=False)
                model, _ = compile_post_search_model(mo_model, model_gen, train_strategy, enable_xla=config.others.enable_XLA_compilation)

            print('Model generated successfully')

            latest = tf.train.latest_checkpoint(os.path.join(m_path, 'weights'))
            model.load_weights(latest)
            print('Weights loaded successfully from checkpoint')

            save_evaluation_results(model, test_ds, m_path)
            # create confusion matrix only in classification tasks
            if config.dataset.type in ['image_classification', 'time_series_classification']:
                predict_and_save_confusion_matrix(model, test_ds, multi_output, n_classes=classes_count,
                                                  save_path=os.path.join(m_path, 'test_confusion_matrix'))

    perform_global_memory_clear()


if __name__ == '__main__':
    main()
