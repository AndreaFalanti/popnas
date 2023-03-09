import argparse
import os
from pathlib import Path

from tensorflow.keras import callbacks
from tensorflow.keras.utils import plot_model

import log_service
from dataset.augmentation import get_image_data_augmentation_model
from dataset.generators.factory import dataset_generator_factory
from dataset.utils import generate_balanced_weights_for_classes
from models.custom_callbacks.training_time import TrainingTimeCallback
from models.generators.factory import model_generator_factory
from models.graphs.utils import save_cell_dag_image
from search_space import CellSpecification
from utils.experiments_summary import FinalTrainingInfo, write_final_training_infos_csv
from utils.nn_utils import save_keras_model_to_onnx, predict_and_save_confusion_matrix, perform_global_memory_clear, \
    remove_annoying_tensorflow_messages
from utils.post_search_training_utils import create_model_log_folder, define_callbacks, \
    save_complete_and_trimmed_json_config, compile_post_search_model, build_config, \
    save_evaluation_results, get_best_cell_specs, extract_final_training_results, log_training_results_summary, \
    log_training_results_dict, extend_keras_metrics

# disable Tensorflow info and warning messages
remove_annoying_tensorflow_messages()


def execute(p: str, b: int, f: int, m: int, n: int, spec: str = None, j: str = None, ts: str = None,
            name: str = 'final_model_training', stem: bool = False):
    ''' Refer to argparse help for more information about these arguments. '''
    save_path = os.path.join(p, name)
    create_model_log_folder(save_path)
    logger = log_service.get_logger(__name__)

    # read final_training config
    custom_json_path = Path(__file__).parent / '../configs/last_training.json' if j is None else j

    logger.info('Reading configuration...')
    config, train_strategy = build_config(p, b, ts, custom_json_path)

    cnn_config = config.cnn_hp
    arc_config = config.architecture_parameters
    ds_config = config.dataset

    multi_output = arc_config.multi_output
    augment_on_gpu = ds_config.data_augmentation.perform_on_gpu
    score_metric_name = config.search_strategy.score_metric

    # override config with command line parameters
    arc_config.motifs = m
    arc_config.normal_cells_per_motif = n
    cnn_config.filters = f

    # Load and prepare the dataset
    logger.info('Preparing datasets...')
    dataset_generator = dataset_generator_factory(ds_config, config.others)
    train_ds, classes_count, input_shape, train_batches, preprocessing_model = dataset_generator.generate_final_training_dataset()
    # produce weights for balanced loss if option is enabled in database config
    balanced_class_weights = generate_balanced_weights_for_classes(train_ds) if ds_config.balance_class_losses else None
    logger.info('Datasets generated successfully')

    # DEBUG ONLY
    # test_data_augmentation(train_ds)

    with train_strategy.scope():
        model_gen = model_generator_factory(ds_config, cnn_config, arc_config, train_batches,
                                            output_classes_count=classes_count, input_shape=input_shape,
                                            data_augmentation_model=get_image_data_augmentation_model() if augment_on_gpu else None,
                                            preprocessing_model=preprocessing_model)

    keras_metrics = model_gen.get_results_processor_class().keras_metrics_considered()
    extended_keras_metrics = extend_keras_metrics(keras_metrics)

    if spec is None:
        logger.info('Getting best cell specification found during POPNAS run...')
        # extract the TargetMetric related to the score metric considered during the search
        target_metric = next(m for m in keras_metrics if m.name == score_metric_name)

        # get the best model found during search and log some relevant info.
        # using n=1, the script obtains the best element, extracted immediately (to avoid using arrays)
        cell_spec, best_score = get_best_cell_specs(p, n=1, metric=target_metric)
        cell_spec, best_score = cell_spec[0], best_score[0]

        logger.info('CELL INFO')
        cell_spec.pretty_logging(logger)
        logger.info('Best score (%s) reached during training: %0.4f', target_metric.name, best_score)
    else:
        cell_spec = CellSpecification.from_str(spec)

    logger.info('Generating Keras model from cell specification...')
    # write cell spec to external file, stored together with results (usable by other scripts and to remember what cell has been trained)
    with open(os.path.join(save_path, 'cell_spec.txt'), 'w') as fp:
        fp.write(str(cell_spec))

    save_complete_and_trimmed_json_config(config, save_path)

    with train_strategy.scope():
        mo_model, output_names = model_gen.build_model(cell_spec, add_imagenet_stem=stem)
        model, output_names = compile_post_search_model(mo_model, model_gen, train_strategy, enable_xla=config.others.enable_XLA_compilation)

    logger.info('Model generated successfully')
    model.summary(line_length=140, print_fn=logger.info)

    logger.info('Converting untrained model to ONNX')
    save_keras_model_to_onnx(model, save_path=os.path.join(save_path, 'untrained.onnx'))

    # Define callbacks
    train_callbacks = define_callbacks(score_metric_name, output_names, use_val=False)
    time_cb = TrainingTimeCallback()
    train_callbacks.insert(0, time_cb)

    plot_model(model, to_file=os.path.join(save_path, 'model.pdf'), show_shapes=True, show_layer_names=True)

    hist = model.fit(x=train_ds,
                     epochs=cnn_config.epochs,
                     steps_per_epoch=train_batches,
                     class_weight=balanced_class_weights,
                     callbacks=train_callbacks)  # type: callbacks.History

    training_time = time_cb.get_total_time()
    results_dict, best_epoch, best_training_score = extract_final_training_results(hist, score_metric_name, extended_keras_metrics,
                                                                                   output_names, using_val=False)
    log_training_results_summary(logger, best_epoch, cnn_config.epochs, training_time, best_training_score, score_metric_name)
    log_training_results_dict(logger, results_dict)

    logger.info('Saving TF model')
    model.save(os.path.join(save_path, 'tf_model'))

    logger.info('Converting trained model to ONNX')
    save_keras_model_to_onnx(model, save_path=os.path.join(save_path, 'trained.onnx'))

    try:
        test_ds, _, _, test_batches = dataset_generator.generate_test_dataset()
        save_evaluation_results(model, test_ds, save_path)
        # create confusion matrix only in classification tasks
        if ds_config.type in ['image_classification', 'time_series_classification']:
            predict_and_save_confusion_matrix(model, test_ds, multi_output, n_classes=classes_count,
                                              save_path=os.path.join(save_path, 'test_confusion_matrix'))
    except:
        logger.info('Could not build the test dataset, or test dataset is not provided')

    info = FinalTrainingInfo(ds_config.name, cell_spec, model.count_params(), m, n, f, best_training_score, training_time, score_metric_name)
    write_final_training_infos_csv(os.path.join(save_path, 'results.csv'), [info])

    perform_global_memory_clear()
    save_cell_dag_image(cell_spec, save_path)


if __name__ == '__main__':
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

    execute(**vars(args))
