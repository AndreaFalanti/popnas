import argparse
import importlib.util

import log_service

import tensorflow as tf
import numpy as np

from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import datasets
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def load_dataset(dataset):
    if dataset == "cifar10":
        (x_train_init, y_train_init), (x_test_init, y_test_init) = datasets.cifar10.load_data()
        y_train_init = to_categorical(y_train_init, 10)
        y_test_init = to_categorical(y_test_init, 10)
    elif dataset == "cifar100":
        (x_train_init, y_train_init), (x_test_init, y_test_init) =  datasets.cifar100.load_data()
        y_train_init = to_categorical(y_train_init, 100)
        y_test_init = to_categorical(y_test_init, 100)
    else:
        spec = importlib.util.spec_from_file_location("dataset", dataset)
        set = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(set)
        (x_train_init, y_train_init), (x_test_init, y_test_init) = set.load_data()

    return (x_train_init, y_train_init), (x_test_init, y_test_init)


def apply_data_augmentation():
    # TODO: convert in command line argument or config file
    apply_data_augmentation = True

    # Create training ImageDataGenerator object
    if apply_data_augmentation:
        train_datagen = ImageDataGenerator(horizontal_flip=True, rescale=1./255)
        validation_datagen = ImageDataGenerator(horizontal_flip=True, rescale=1./255)
    else:
        train_datagen = ImageDataGenerator()
        validation_datagen = ImageDataGenerator()

    return train_datagen, validation_datagen


def define_callbacks():
    '''
    Define callbacks used in model training.

    Returns:
        (tf.keras.Callback[]): Keras callbacks
    '''
    callbacks = []

    # Save best weights
    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=log_service.build_path('weights', 'cp_e{epoch:02d}_vl{val_loss:.2f}.ckpt'),
                                                        save_weights_only=True, save_best_only=True, monitor='val_loss', mode='min')
    callbacks.append(ckpt_callback)
    
    # By default shows losses and metrics for both training and validation
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_service.build_path('tensorboard'), profile_batch=0, histogram_freq=0)

    callbacks.append(tb_callback)

    # TODO: convert into a parameter
    early_stop = True
    if early_stop:
        es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1)
        callbacks.append(es_callback)

    return callbacks


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-p', metavar=('PATH'), type=str, help="path to best model folder", required=True)
    args = parser.parse_args()

    log_service.initialize_log_folders_best_model_script()
    logger = log_service.get_logger(__name__)

    logger.info('Loading best model from provided folder...')
    model = keras.models.load_model(args.p)
    model.summary(line_length=140, print_fn=logger.info)

    # Load and prepare the dataset
    logger.info('Preparing dataset...')
    (x_train_init, y_train_init), (x_test_init, y_test_init) = load_dataset('cifar10')
    x_train, x_validation, y_train, y_validation = train_test_split(x_train_init, y_train_init, train_size=0.8, shuffle=True) # use only 80% of the samples

    bs = 128 # TODO: batch size, make it a parameter?

    # TODO: replaced by datagen method below, if it works fine then delete this
    # train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    # validation_dataset = tf.data.Dataset.from_tensor_slices((x_validation, y_validation))

    # train_dataset = train_dataset.batch(bs)
    # train_dataset = train_dataset.repeat()

    # validation_dataset = validation_dataset.batch(bs)
    # validation_dataset = validation_dataset.repeat()

    train_datagen, validation_datagen = apply_data_augmentation()
    train_datagen.fit(x_train)
    validation_datagen.fit(x_validation)

    train_dataset = train_datagen.flow(x_train, y_train, batch_size=bs)
    validation_dataset = validation_datagen.flow(x_validation, y_validation, batch_size=bs)


    # Define training procedure and hyperparameters
    loss = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    metrics = ['accuracy',  tf.keras.metrics.TopKCategoricalAccuracy(k=5)]

    # Compile model (should also reinitialize the weights, providing training from scratch)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Define callbacks
    callbacks = define_callbacks()

    model.fit(x=train_dataset,
                epochs=300,
                batch_size=128,
                steps_per_epoch=np.ceil(len(x_train) / bs),
                validation_data=validation_dataset,
                validation_steps=np.ceil(len(x_validation) / bs),
                callbacks=callbacks)


if __name__ == '__main__':
    main()