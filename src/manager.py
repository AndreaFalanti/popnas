import log_service

from tqdm import tqdm
import numpy as np

import os
import shutil

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils.vis_utils import plot_model   # per stampare modello

from model import ModelGenerator
from utils.TimingCallback import TimingCallback


if not os.path.exists('temp_weights/'):
    os.makedirs('temp_weights/')
else:
    shutil.rmtree('temp_weights')
    os.makedirs('temp_weights/', exist_ok=True)


class NetworkManager:
    '''
    Helper class to manage the generation of subnetwork training given a dataset
    '''

    def __init__(self, dataset, data_num=1, epochs=20, batchsize=64, learning_rate=0.01, filters=24):
        '''
        Manager which is tasked with creating subnetworks, training them on a dataset, and retrieving
        rewards in the term of accuracy, which is passed to the controller RNN.

        # Args:
            dataset: a tuple of 4 arrays (X_train, y_train, X_val, y_val)
            epochs: number of epochs to train the subnetworks
            batchsize: batchsize
            learning_rate: learning rate for the Optimizer
            filters (int): initial number of filters
        '''
        self._logger = log_service.get_logger(__name__)

        self.data_num = data_num
        self.dataset = dataset
        self.epochs = epochs
        self.batchsize = batchsize
        self.lr = learning_rate
        self.filters = filters

        self.num_child = 0  # SUMMARY
        self.best_reward = 0.0

    def __compile_model(self, model_fn: ModelGenerator, actions: 'list[str]', concat_only_unused: bool, tb_logdir: str):
        '''
        Generate and compile a Keras model, with cell structure defined by actions provided.

        Args:
            model_fn (ModelGenerator): [description]
            actions (list): [description]
            concat_only_unused (bool): [description]
            tb_logdir (str): path for tensorboard logging

        Returns:
            (tf.keras.Model, list(tf.keras.callbacks.Callback)): model and callbacks to use while training
        '''
        model_gen = model_fn(actions, self.filters, concat_only_unused)  # type: ModelGenerator
        model = model_gen.build_model()

        loss, optimizer, metrics = model_gen.define_training_hyperparams_and_metrics(self.lr)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        return model, model_gen.define_callbacks(tb_logdir)

    def __build_datasets(self, dataset_index: int, use_data_augmentation: bool):
        '''
        Build the training and validation datasets to be used in model.fit().

        Args:
            dataset_index (int): dataset index
            use_data_augmentation (bool): [description]

        Returns:
            [type]: [description]
        '''

        x_train, y_train, x_val, y_val = self.dataset[dataset_index]

        if use_data_augmentation:
            train_datagen = ImageDataGenerator(horizontal_flip=True)
            validation_datagen = ImageDataGenerator(horizontal_flip=True)
        else:
            train_datagen = ImageDataGenerator()
            validation_datagen = ImageDataGenerator()

        train_datagen.fit(x_train)
        validation_datagen.fit(x_val)

        train_dataset = train_datagen.flow(x_train, y_train, batch_size=self.batchsize)
        validation_dataset = validation_datagen.flow(x_val, y_val, batch_size=self.batchsize)

        train_batches = np.ceil(len(x_train) / self.batchsize)
        val_batches = np.ceil(len(x_val) / self.batchsize)

        return train_dataset, validation_dataset, train_batches, val_batches

    def get_rewards(self, model_fn, actions, concat_only_unused=True, save_best_model=False, display_model_summary=True):
        '''
        Creates a CNN given the actions predicted by the controller RNN,
        trains it on the provided dataset, and then returns a reward.

        # Args:
            model_fn: a function which accepts one argument, a list of
                parsed actions, obtained via an inverse mapping from the
                StateSpace.

            actions: a list of parsed actions obtained via an inverse mapping
                from the StateSpace. It is in a specific order as given below:

                Consider 4 states were added to the StateSpace via the `add_state`
                method. Then the `actions` array will be of length 4, with the
                values of those states in the order that they were added.

                If number of layers is greater than one, then the `actions` array
                will be of length `4 * number of layers` (in the above scenario).
                The index from [0:4] will be for layer 0, from [4:8] for layer 1,
                etc for the number of layers.

                These action values are for direct use in the construction of models.

            concat_only_unused (bool): concat only unused block outputs in the cell output

            display_model_summary: Display the child model summary at the end of training.

        # Returns:
            a reward for training a model with the given actions
        '''

        tf.keras.backend.reset_uids()

        # create children folder on Tensorboard
        self.num_child = self.num_child + 1
        # grouped for block count and enumerated progressively
        tb_logdir = log_service.build_path('tensorboard_cnn', f'B{len(actions) // 4}', str(self.num_child))

        # generate a submodel given predicted actions
        for index in range(self.data_num):
            if self.data_num > 1:
                self._logger.info("Training dataset #%d / #%d", index + 1, self.data_num)

            # build the model, given the actions
            model, callbacks = self.__compile_model(model_fn, actions, concat_only_unused, tb_logdir)
            # add callback to register as accurate as possible the training time
            time_cb = TimingCallback()
            callbacks.append(time_cb)

            train_ds, val_ds, train_batches, val_batches = self.__build_datasets(index, True)

            hist = model.fit(x=train_ds,
                        epochs=self.epochs,
                        batch_size=self.batchsize,
                        steps_per_epoch=train_batches,
                        validation_data=val_ds,
                        validation_steps=val_batches,
                        callbacks=callbacks)

            timer = sum(time_cb.logs)

            # display the structure of the child model
            if display_model_summary:
                model.summary(line_length=140, print_fn=self._logger.info)
                # plot_model(model, to_file='%s/model_plot.png' % self.logdir, show_shapes=True, show_layer_names=True)

        # compute the reward (best validation accuracy)
        reward = max(hist.history.get('val_accuracy'))

        self._logger.info("Manager: Accuracy = %0.6f, Training time: %0.6f seconds", reward, timer)

        # if algorithm is training the last models batch (B = value provided in command line)
        # save the best model in a folder, so that can be trained from scratch later on
        if save_best_model and reward > self.best_reward:
            self.best_reward = reward
            # last model should be automatically overwritten, leaving only one model
            self._logger.info('Saving model...')
            model.save(log_service.build_path('best_model'))
            self._logger.info('Model saved successfully')


        # clean up resources and GPU memory
        del model

        return [reward, timer]
