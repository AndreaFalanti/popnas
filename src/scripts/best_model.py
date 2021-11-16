import argparse
import importlib.util
import operator
import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import datasets, callbacks, optimizers, losses, models, metrics
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

import log_service
from utils.func_utils import create_empty_folder
from utils.timing_callback import TimingCallback

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # disable Tensorflow info messages


def create_log_folder(log_path: str):
    model_training_folder_path = os.path.join(log_path, 'best_model_training')
    create_empty_folder(model_training_folder_path)
    os.mkdir(os.path.join(model_training_folder_path, 'weights'))  # create weights folder
    os.mkdir(os.path.join(model_training_folder_path, 'tensorboard'))  # create tensorboard folder

    log_service.set_log_path(model_training_folder_path)
    return model_training_folder_path


def load_dataset(dataset):
    if dataset == "cifar10":
        (x_train_init, y_train_init), (x_test_init, y_test_init) = datasets.cifar10.load_data()
        y_train_init = to_categorical(y_train_init, 10)
        y_test_init = to_categorical(y_test_init, 10)
    elif dataset == "cifar100":
        (x_train_init, y_train_init), (x_test_init, y_test_init) = datasets.cifar100.load_data()
        y_train_init = to_categorical(y_train_init, 100)
        y_test_init = to_categorical(y_test_init, 100)
    # TODO: untested legacy code, not sure this is still working
    else:
        spec = importlib.util.spec_from_file_location("dataset", dataset)
        dataset = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dataset)
        (x_train_init, y_train_init), (x_test_init, y_test_init) = dataset.load_data()

    return (x_train_init, y_train_init), (x_test_init, y_test_init)


def apply_data_augmentation():
    # TODO: convert in command line argument or config file
    use_data_augmentation = True

    # Create training ImageDataGenerator object
    if use_data_augmentation:
        train_datagen = ImageDataGenerator(horizontal_flip=True,
                                           rotation_range=20,
                                           width_shift_range=4,
                                           height_shift_range=4,
                                           zoom_range=0.1,
                                           rescale=1. / 255)
    else:
        train_datagen = ImageDataGenerator()

    validation_datagen = ImageDataGenerator()

    return train_datagen, validation_datagen


def generate_dataset(batch_size: int):
    (x_train_init, y_train_init), _ = load_dataset('cifar10')
    x_train, x_val, y_train, y_val = train_test_split(x_train_init, y_train_init, train_size=0.9, shuffle=True, stratify=y_train_init)

    train_datagen, validation_datagen = apply_data_augmentation()
    train_datagen.fit(x_train)
    validation_datagen.fit(x_val)

    cifar10_signature = (tf.TensorSpec(shape=(None, 32, 32, 3), dtype=tf.float32),
                         tf.TensorSpec(shape=(None, 10), dtype=tf.float32))
    train_dataset = tf.data.Dataset.from_generator(train_datagen.flow, args=(x_train, y_train, batch_size), output_signature=cifar10_signature)
    validation_dataset = tf.data.Dataset.from_generator(train_datagen.flow, args=(x_val, y_val, batch_size), output_signature=cifar10_signature)

    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)

    train_batches = np.ceil(len(x_train) / batch_size)
    val_batches = np.ceil(len(x_val) / batch_size)

    return train_dataset, validation_dataset, train_batches, val_batches


def define_callbacks() -> 'list[callbacks.Callback]':
    '''
    Define callbacks used in model training.

    Returns:
        (tf.keras.Callback[]): Keras callbacks
    '''
    # Save best weights
    ckpt_save_format = 'cp_e{epoch:02d}_vl{val_loss:.2f}_vacc{val_accuracy:.4f}.ckpt'
    ckpt_callback = callbacks.ModelCheckpoint(filepath=log_service.build_path('weights', ckpt_save_format),
                                              save_weights_only=True, save_best_only=False, monitor='val_accuracy', mode='max')
    # By default shows losses and metrics for both training and validation
    tb_callback = callbacks.TensorBoard(log_dir=log_service.build_path('tensorboard'), profile_batch=0, histogram_freq=1)

    es_callback = callbacks.EarlyStopping(monitor='val_accuracy', patience=12, restore_best_weights=True, verbose=1, mode='max')
    plateau_callback = callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.4, patience=5, verbose=1, mode='max')

    return [ckpt_callback, tb_callback, es_callback, plateau_callback]


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-p', metavar='PATH', type=str, help="path to log folder", required=True)
    args = parser.parse_args()

    save_path = create_log_folder(args.p)
    logger = log_service.get_logger(__name__)

    logger.info('Loading best model from provided folder...')
    model = models.load_model(os.path.join(args.p, 'best_model'))  # type: models.Model
    model.summary(line_length=140, print_fn=logger.info)

    # Load and prepare the dataset
    logger.info('Preparing datasets...')
    train_dataset, validation_dataset, train_batches, val_batches = generate_dataset(batch_size=128)

    # Define training procedure and hyperparameters
    loss = losses.CategoricalCrossentropy()
    optimizer = optimizers.Adam(learning_rate=0.01)
    train_metrics = ['accuracy', metrics.TopKCategoricalAccuracy(k=3)]

    # Compile model (should also reinitialize the weights, providing training from scratch)
    model.compile(optimizer=optimizer, loss=loss, metrics=train_metrics)

    # Define callbacks
    train_callbacks = define_callbacks()
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
    loss, acc, top3 = hist.history['loss'][epoch_index], hist.history['accuracy'][epoch_index],\
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
