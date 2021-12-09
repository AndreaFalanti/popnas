import argparse
import importlib.util
import json
import operator
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import datasets, callbacks, optimizers, losses, models, metrics, layers, Sequential
from tensorflow.keras.utils import to_categorical

import log_service
from model import ModelGenerator
from utils.func_utils import create_empty_folder, parse_cell_structures
from utils.rstr import rstr
from utils.timing_callback import TimingCallback

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # disable Tensorflow info messages

AUTOTUNE = tf.data.AUTOTUNE


def create_log_folder(log_path: str):
    model_training_folder_path = os.path.join(log_path, 'best_model_training')
    create_empty_folder(model_training_folder_path)
    os.mkdir(os.path.join(model_training_folder_path, 'weights'))  # create weights folder
    os.mkdir(os.path.join(model_training_folder_path, 'tensorboard'))  # create tensorboard folder

    log_service.set_log_path(model_training_folder_path)
    return model_training_folder_path


def get_best_cell_spec(log_folder_path: str):
    training_results_csv_path = os.path.join(log_folder_path, 'csv', 'training_results.csv')
    df = pd.read_csv(training_results_csv_path)
    best_acc_row = df.loc[df['best val accuracy'].idxmax()]

    cell_spec = parse_cell_structures([best_acc_row['cell structure']])[0]
    return cell_spec, best_acc_row['best val accuracy']


def load_run_json(log_folder_path: str):
    json_path = os.path.join(log_folder_path, 'restore', 'run.json')
    with open(json_path, 'r') as f:
        run_config = json.load(f)

    return run_config


def load_dataset(dataset):
    if dataset == "cifar10":
        (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
        classes_count = 10
    elif dataset == "cifar100":
        (x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()
        classes_count = 100
    # TODO: untested legacy code, not sure this is still working
    else:
        spec = importlib.util.spec_from_file_location("dataset", dataset)
        dataset = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dataset)
        (x_train, y_train), (x_test, y_test) = dataset.load_data()
        classes_count = None

    # preprocessing step
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    y_train = to_categorical(y_train, classes_count)
    y_test = to_categorical(y_test, classes_count)

    return (x_train, y_train), (x_test, y_test), classes_count


def generate_dataset(batch_size: int):
    (x_train_init, y_train_init), _, classes_count = load_dataset('cifar10')
    x_train, x_val, y_train, y_val = train_test_split(x_train_init, y_train_init, train_size=0.9, shuffle=True, stratify=y_train_init)

    # follow similar augmentation techniques used in other papers, which usually are:
    # - horizontal flip
    # - 4px translate on both height and width [fill=reflect] (sometimes upscale to 40x40, with random crop to original 32x32)
    # - whitening (not always used)
    data_augmentation = Sequential([
        layers.experimental.preprocessing.RandomFlip('horizontal'),
        # layers.experimental.preprocessing.RandomRotation(20/360),   # 20 degrees range
        # layers.experimental.preprocessing.RandomZoom(height_factor=0.1, width_factor=0.1),
        layers.experimental.preprocessing.RandomTranslation(height_factor=0.125, width_factor=0.125)
    ], name='data_augmentation')

    # create a batched dataset, cached in memory for better performance
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size).cache()
    validation_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size).cache().prefetch(AUTOTUNE)

    # perform data augmentation
    train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    train_batches = np.ceil(len(x_train) / batch_size)
    val_batches = np.ceil(len(x_val) / batch_size)

    return train_dataset, validation_dataset, int(train_batches), int(val_batches), classes_count


def define_callbacks(cdr_enabled: bool) -> 'list[callbacks.Callback]':
    '''
    Define callbacks used in model training.

    Returns:
        (tf.keras.Callback[]): Keras callbacks
    '''
    # Save best weights
    ckpt_save_format = 'cp_e{epoch:02d}_vl{val_loss:.2f}_vacc{val_accuracy:.4f}.ckpt'
    ckpt_callback = callbacks.ModelCheckpoint(filepath=log_service.build_path('weights', ckpt_save_format),
                                              save_weights_only=True, save_best_only=True, monitor='val_accuracy', mode='max')
    # By default shows losses and metrics for both training and validation
    tb_callback = callbacks.TensorBoard(log_dir=log_service.build_path('tensorboard'), profile_batch=0, histogram_freq=0)

    es_callback = callbacks.EarlyStopping(monitor='val_accuracy', patience=12, restore_best_weights=True, verbose=1, mode='max')

    # these callbacks are shared between all models
    train_callbacks = [ckpt_callback, tb_callback, es_callback]

    # if using plain lr, adapt it with reduce learning rate on plateau
    # NOTE: for unknown reasons, activating plateau callback when cdr is present will also cause an error at the end of the first epoch
    if not cdr_enabled:
        train_callbacks.append(callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.4, patience=5, verbose=1, mode='max'))

    return [ckpt_callback, tb_callback, es_callback]


def main():
    # spec argument can be taken from model summary.txt, changing commas between tuples with ;
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-p', metavar='PATH', type=str, help="path to log folder", required=True)
    parser.add_argument('-spec', metavar='CELL_SPECIFICATION', type=str, help="cell specification string", default=None)
    parser.add_argument('--load', help='load model from checkpoint', action='store_true')
    parser.add_argument('--same', help='use same hyperparams of the ones used during search algorithm', action='store_true')
    args = parser.parse_args()

    save_path = create_log_folder(args.p)
    logger = log_service.get_logger(__name__)

    # Load and prepare the dataset
    logger.info('Preparing datasets...')
    train_dataset, validation_dataset, train_batches, val_batches, classes_count = generate_dataset(batch_size=128)
    logger.info('Datasets generated successfully')

    # load model from checkpoint
    if args.load:
        logger.info('Loading best model from provided folder...')
        model = models.load_model(os.path.join(args.p, 'best_model'))  # type: models.Model

        # Define training procedure and hyperparameters
        loss = losses.CategoricalCrossentropy()
        optimizer = optimizers.Adam(learning_rate=0.01)
        train_metrics = ['accuracy', metrics.TopKCategoricalAccuracy(k=3)]

        # Compile model (should also reinitialize the weights, providing training from scratch)
        model.compile(optimizer=optimizer, loss=loss, metrics=train_metrics)

        cdr_enabled = False
        logger.info('Model loaded successfully')
    else:
        if args.spec is None:
            # find best model found during search and log some relevant info
            cell_spec, best_acc = get_best_cell_spec(args.p)
            logger.info('%s', '*' * 22 + ' BEST CELL INFO ' + '*' * 22)
            logger.info('Cell specification:')
            for i, block in enumerate(cell_spec):
                logger.info("Block %d: %s", i + 1, rstr(block))
            logger.info('Best validation accuracy reached during training: %0.4f', best_acc)
            logger.info('*' * 60)

            logger.info('Generating Keras model from best cell specification...')
        else:
            cell_spec = parse_cell_structures([args.spec])[0]
            logger.info('Generating Keras model from given cell specification...')

        # reproduce the model from run configuration
        config = load_run_json(args.p)
        cnn_config = config['cnn_hp']
        arc_config = config['architecture_parameters']

        if not args.same:
            # optimize some hyperparameters for final training
            config['cnn_hp']['weight_reg'] = 1e-4
            config['cnn_hp']['use_adamW'] = True
            # config['cnn_hp']['drop_path_prob'] = 0.2
            config['cnn_hp']['drop_path_prob'] = 0.0

            config['cnn_hp']['cosine_decay_restart'] = {
                "enabled": True,
                "period_in_epochs": 2,
                "t_mul": 2.0,
                "m_mul": 0.9,
                "alpha": 0.0
            }

        model_gen = ModelGenerator(cnn_config, arc_config, train_batches, output_classes=classes_count, data_augmentation_model=None)
        model, _ = model_gen.build_model(cell_spec)

        loss, loss_weights, optimizer, train_metrics = model_gen.define_training_hyperparams_and_metrics()
        train_metrics.append(metrics.TopKCategoricalAccuracy(k=3))

        model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights, metrics=train_metrics)

        cdr_enabled = cnn_config['cosine_decay_restart']['enabled']
        logger.info('Model generated successfully')

    model.summary(line_length=140, print_fn=logger.info)

    # Define callbacks
    train_callbacks = define_callbacks(cdr_enabled)
    time_cb = TimingCallback()
    train_callbacks.append(time_cb)

    hist = model.fit(x=train_dataset,
                     epochs=300,
                     steps_per_epoch=train_batches,
                     validation_data=validation_dataset,
                     validation_steps=val_batches,
                     callbacks=train_callbacks)

    training_time = sum(time_cb.logs)

    # hist.history is a dictionary of lists (each metric is a key)
    epoch_index, best_val_accuracy = max(enumerate(hist.history['val_accuracy']), key=operator.itemgetter(1))
    loss, acc, top3 = hist.history['loss'][epoch_index], hist.history['accuracy'][epoch_index], \
                      hist.history['top_k_categorical_accuracy'][epoch_index]
    val_loss, top3_val = hist.history['val_loss'][epoch_index], hist.history['val_top_k_categorical_accuracy'][epoch_index]

    logger.info('*' * 60)
    logger.info('Best epoch (val_accuracy) stats:')
    logger.info('Best epoch index: %d', epoch_index + 1)
    logger.info('Total epochs: %d', len(hist.history['loss']))
    logger.info('Total training time (without callbacks): %0.4f seconds', training_time)
    logger.info('-' * 24 + ' Validation ' + '-' * 24)
    logger.info('Validation accuracy: %0.4f', best_val_accuracy)
    logger.info('Validation loss: %0.4f', val_loss)
    logger.info('Validation top3 accuracy: %0.4f', top3_val)
    logger.info('-' * 25 + ' Training ' + '-' * 25)
    logger.info('Accuracy: %0.4f', acc)
    logger.info('Loss: %0.4f', loss)
    logger.info('Top3 accuracy: %0.4f', top3)
    logger.info('*' * 60)
    logger.info('Total training time (without callbacks): %0.4f seconds (%d hours %d minutes %d seconds)',
                training_time, training_time // 3600, (training_time // 60) % 60, training_time % 60)


if __name__ == '__main__':
    main()
