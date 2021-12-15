import numpy as np
import shutil
import os
from tqdm import tqdm

import time
import csv

import tensorflow as tf

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import ModelCheckpoint

from keras.utils.vis_utils import plot_model  # per stampare modello

if not os.path.exists('temp_weights/'):
    os.makedirs('temp_weights/')
else:
    shutil.rmtree('temp_weights')
    os.makedirs('temp_weights/', exist_ok=True)


class NetworkManager:
    '''
    Helper class to manage the generation of subnetwork training given a dataset
    '''

    def __init__(self, dataset, timestr, data_num=1, epochs=5, batchsize=64, learning_rate=0.002):
        '''
        Manager which is tasked with creating subnetworks, training them on a dataset, and retrieving
        rewards in the term of accuracy, which is passed to the controller RNN.

        # Args:
            dataset: a tuple of 4 arrays (X_train, y_train, X_val, y_val)
            epochs: number of epochs to train the subnetworks
            batchsize: batchsize of training the subnetworks
            learning_rate: learning rate for the Optimizer.
        '''
        self.data_num = data_num
        self.dataset = dataset
        self.timestr = timestr
        self.epochs = epochs
        self.batchsize = batchsize
        self.lr = learning_rate

        self.num_child = 0  # SUMMARY

    def get_rewards(self, model_fn, actions, display_model_summary=True):
        '''
        Creates a subnetwork given the actions predicted by the controller RNN,
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

            display_model_summary: Display the child model summary at the end of training.

        # Returns:
            a reward for training a model with the given actions
        '''
        '''
        if tf.test.is_gpu_available():
            device = '/gpu:0'
        else:
            device = '/cpu:0'
        '''

        device = '/gpu:0'
        tf.keras.backend.reset_uids()

        # create children folder on Tensorboard
        self.num_child = self.num_child + 1
        self.logdir = 'logs/%s/children/%s' % (self.timestr, self.num_child)
        self.summary_writer = tf.summary.create_file_writer(self.logdir)
        self.summary_writer.set_as_default()
        tf.name_scope("children")

        acc_list = []

        # generate a submodel given predicted actions
        with tf.device(device):
            for index in range(self.data_num):
                if self.data_num > 1:
                    print("\nTraining dataset #%d / #%d" % (index + 1, self.data_num))
                model = model_fn(actions)  # type: Model

                # build model shapes
                x_train, y_train, x_val, y_val = self.dataset[index]

                # generate the dataset for training
                train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
                    .shuffle(buffer_size=10000, seed=0, reshuffle_each_iteration=True) \
                    .batch(self.batchsize)
                train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
                # train_dataset = train_dataset.apply(tf.data.experimental.prefetch_to_device(device)) # dà errore su GPU

                # generate the dataset for evaluation
                val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(self.batchsize)
                val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
                # val_dataset = val_dataset.apply(tf.data.experimental.prefetch_to_device(device)) # dà errore su GPU

                num_train_batches = x_train.shape[0] // self.batchsize + 1

                train_loss = tf.keras.losses.CategoricalCrossentropy(name='train_loss')

                val_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

                global_step = tf.compat.v1.train.get_or_create_global_step()
                lr = tf.compat.v1.train.cosine_decay(self.lr, global_step, decay_steps=num_train_batches * self.epochs, alpha=0.1)

                # construct the optimizer and saver of the child model
                optimizer = tf.compat.v1.train.AdamOptimizer(lr)
                saver = tf.train.Checkpoint(model=model, optimizer=optimizer, global_step=global_step)

                best_val_acc = 0.0
                timer = 0  # inizialize timer to evaluate training time

                print()

                for epoch in range(self.epochs):
                    # train child model
                    with tqdm(train_dataset,
                              desc='Train Epoch (%d / %d): ' % (epoch + 1, self.epochs),
                              total=num_train_batches) as iterator:

                        for i, (x, y) in enumerate(iterator):
                            # get gradients
                            with tf.GradientTape() as tape:

                                # get training starting time
                                start = time.clock()
                                preds = model(x, training=True)
                                loss = train_loss(y, preds)

                            grad = tape.gradient(loss, model.trainable_variables)
                            grad_vars = zip(grad, model.trainable_variables)

                            # update weights of the child models
                            optimizer.apply_gradients(grad_vars, global_step)

                            # get training ending time
                            stop = time.clock()

                            # evaluate training time
                            timer = timer + (stop - start)

                            if (i + 1) >= num_train_batches:
                                break

                    print()

                    # evaluate child model
                    for j, (x, y) in enumerate(val_dataset):
                        preds = model(x, training=False)
                        val_accuracy(y, preds)

                    acc = val_accuracy.result().numpy()

                    # add forward pass and accuracy to Tensorboard
                    tf.summary.scalar("training_time", timer, step=epoch + 1)
                    if acc > best_val_acc:
                        summary_acc = acc
                    else:
                        summary_acc = best_val_acc
                    tf.summary.scalar("child_accuracy", summary_acc, step=epoch + 1)

                    print("\tEpoch %d: Training time = %0.6f" % (epoch + 1, timer))
                    print("\tEpoch %d: Val accuracy = %0.6f" % (epoch + 1, acc))

                    # if acc improved, save the weights
                    if acc > best_val_acc:
                        print("\tVal accuracy improved from %0.6f to %0.6f. Saving weights !" % (
                            best_val_acc, acc))

                        best_val_acc = acc
                        saver.save('temp_weights/temp_network')

                    print()

                # test_writer.close()

                # load best weights of the child model
                path = tf.train.latest_checkpoint('temp_weights/')
                saver.restore(path)

                # display the structure of the child model
                if display_model_summary:
                    model.summary()
                    # plot_model(model, to_file='%s/model_plot.png' % self.logdir, show_shapes=True, show_layer_names=True)

                for j, (x, y) in enumerate(val_dataset):
                    preds = model(x, training=False)
                    val_accuracy(y, preds)

                acc = val_accuracy.result().numpy()
                acc_list.append(acc)

            # compute the reward (validation accuracy)
            reward = acc

            print()
            print("Manager: Accuracy = ", reward)

        # clean up resources and GPU memory

        del model
        del optimizer
        del global_step

        return [reward, timer]
