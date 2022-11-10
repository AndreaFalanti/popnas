import argparse
import logging
import os
from pathlib import Path

import pandas as pd
import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.python.keras.utils.vis_utils import plot_model

import log_service
from dataset.augmentation import get_image_data_augmentation_model
from dataset.utils import dataset_generator_factory, generate_balanced_weights_for_classes
from models.model_generator import ModelGenerator
from utils.feature_utils import metrics_fields_dict
from utils.func_utils import parse_cell_structures, cell_spec_to_str
from utils.network_graph import save_cell_dag_image
from utils.nn_utils import save_keras_model_to_onnx, predict_and_save_confusion_matrix
from utils.post_search_training_utils import create_model_log_folder, log_best_cell_results_during_search, define_callbacks, \
    log_final_training_results, override_checkpoint_callback, save_trimmed_json_config, compile_post_search_model, build_config, \
    save_evaluation_results
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
    parser.add_argument('-b', metavar='BATCH SIZE', type=int, help="desired batch size", required=True)
    parser.add_argument('-f', metavar='FILTERS', type=int, help="desired starting filters", required=True)
    parser.add_argument('-m', metavar='MOTIFS', type=int, help="desired motifs", required=True)
    parser.add_argument('-n', metavar='NORMAL CELLS PER MOTIF', type=int, help="desired normal cells per motif", required=True)
    parser.add_argument('-ts', metavar='TRAIN_STRATEGY', type=str, help='device used in Tensorflow distribute strategy', default=None)
    parser.add_argument('-spec', metavar='CELL_SPECIFICATION', type=str, help="cell specification string", default=None)
    parser.add_argument('-name', metavar='OUTPUT_NAME', type=str, help="output location in log folder", default='final_model_training')
    parser.add_argument('--stem', help='add ImageNet stem to network architecture', action='store_true')
    args = parser.parse_args()

    save_path = os.path.join(args.p, args.name)
    create_model_log_folder(save_path)
    logger = log_service.get_logger(__name__)

    # read final_training config
    custom_json_path = Path(__file__).parent / '../configs/last_training.json' if args.j is None else args.j

    logger.info('Reading configuration...')
    config, train_strategy = build_config(args, custom_json_path)

    cnn_config = config['cnn_hp']
    arc_config = config['architecture_parameters']

    multi_output = arc_config['multi_output']
    augment_on_gpu = config['dataset']['data_augmentation']['perform_on_gpu']
    score_metric = config['search_strategy']['score_metric']

    arc_config['motifs'] = args.m
    arc_config['normal_cells_per_motif'] = args.n
    cnn_config['filters'] = args.f

    # Load and prepare the dataset
    logger.info('Preparing datasets...')
    dataset_generator = dataset_generator_factory(config['dataset'])
    train_ds, classes_count, input_shape, train_batches, preprocessing_model = dataset_generator.generate_final_training_dataset()
    test_ds, _, _, test_batches = dataset_generator.generate_test_dataset()

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

    save_trimmed_json_config(config, save_path)

    with train_strategy.scope():
        model_gen = ModelGenerator(cnn_config, arc_config, train_batches, output_classes_count=classes_count, input_shape=input_shape,
                                   data_augmentation_model=get_image_data_augmentation_model() if augment_on_gpu else None,
                                   preprocessing_model=preprocessing_model)

        mo_model, _, last_cell_index = model_gen.build_model(cell_spec, add_imagenet_stem=args.stem)
        model = compile_post_search_model(mo_model, model_gen, train_strategy)

    logger.info('Model generated successfully')

    model.summary(line_length=140, print_fn=logger.info)

    logger.info('Converting untrained model to ONNX')
    save_keras_model_to_onnx(model, save_path=os.path.join(save_path, 'untrained.onnx'))

    # Define callbacks
    train_callbacks = define_callbacks(score_metric, multi_output, last_cell_index)
    override_checkpoint_callback(train_callbacks, score_metric, last_cell_index, use_val=False)
    time_cb = TimingCallback()
    train_callbacks.insert(0, time_cb)

    plot_model(model, to_file=os.path.join(save_path, 'model.pdf'), show_shapes=True, show_layer_names=True)

    hist = model.fit(x=train_ds,
                     epochs=cnn_config['epochs'],
                     steps_per_epoch=train_batches,
                     class_weight=balanced_class_weights,
                     callbacks=train_callbacks)     # type: callbacks.History

    training_time = sum(time_cb.logs)
    log_final_training_results(logger, hist, score_metric, training_time, arc_config['multi_output'], using_val=False)

    logger.info('Saving TF model')
    model.save(os.path.join(save_path, 'tf_model'))

    logger.info('Converting trained model to ONNX')
    save_keras_model_to_onnx(model, save_path=os.path.join(save_path, 'trained.onnx'))

    save_evaluation_results(model, test_ds, save_path)
    predict_and_save_confusion_matrix(model, test_ds, multi_output, n_classes=classes_count,
                                      save_path=os.path.join(save_path, 'test_confusion_matrix'))
    save_cell_dag_image(cell_spec, save_path)


if __name__ == '__main__':
    main()
