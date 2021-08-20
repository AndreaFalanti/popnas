import log_service
from keras.utils.vis_utils import plot_model  # per stampare modello
from tensorflow.keras.models import Model
import shutil
import os
from tqdm import tqdm

import time

import tensorflow as tf
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False  # to hide warning
# tf.compat.v1.enable_eager_execution(device_policy=tf.contrib.eager.DEVICE_PLACEMENT_SILENT)


if not os.path.exists('temp_weights/'):
    os.makedirs('temp_weights/')
else:
    shutil.rmtree('temp_weights')
    os.makedirs('temp_weights/', exist_ok=True)


class NetworkManager:
    '''
    Helper class to manage the generation of subnetwork training given a dataset
    '''

    def __init__(self, dataset, data_num=1, epochs=20, batchsize=64, learning_rate=0.01, filters=24, cpu=False):
        '''
        Manager which is tasked with creating subnetworks, training them on a dataset, and retrieving
        rewards in the term of accuracy, which is passed to the controller RNN.

        # Args:
            dataset: a tuple of 4 arrays (X_train, y_train, X_val, y_val)
            epochs: number of epochs to train the subnetworks
            batchsize: batchsize of training the subnetworks
            learning_rate: learning rate for the Optimizer
            filters (int): initial number of filters
            cpu (bool): use CPU for training networks
        '''
        self._logger = log_service.get_logger(__name__)

        self.data_num = data_num
        self.dataset = dataset
        self.epochs = epochs
        self.batchsize = batchsize
        self.lr = learning_rate
        self.filters = filters
        self.cpu = cpu

        self.num_child = 0  # SUMMARY
        self.best_reward = 0.0

    def get_rewards(self, model_fn, actions, concat_only_unused=True, save_best_model=False, display_model_summary=True):
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

            concat_only_unused (bool): concat only unused block outputs in the cell output

            display_model_summary: Display the child model summary at the end of training.

        # Returns:
            a reward for training a model with the given actions
        '''

        device = '/cpu:0' if self.cpu else '/gpu:0'
        tf.keras.backend.reset_uids()

        # create children folder on Tensorboard
        self.num_child = self.num_child + 1
        self.logdir = log_service.build_path('children', str(self.num_child))
        self.summary_writer = tf.summary.create_file_writer(self.logdir)
        self.summary_writer.set_as_default()

        acc_list = []

        # generate a submodel given predicted actions
        with tf.device(device):
            for index in range(self.data_num):
                if self.data_num > 1:
                    self._logger.info("Training dataset #%d / #%d", index + 1, self.data_num)

                # build the model, given the actions
                model = model_fn(actions, self.filters, concat_only_unused).model  # type: Model

                # build model shapes
                x_train, y_train, x_val, y_val = self.dataset[index]

                # TODO: seems that repeat is not necessary
                # generate the dataset for training
                train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
                train_dataset = train_dataset.shuffle(10000, seed=0)
                train_dataset = train_dataset.batch(self.batchsize)
                #train_dataset = train_dataset.repeat()
                train_dataset = train_dataset.prefetch(4)  # da usare se prefetch_to_device non funziona
                # train_dataset = train_dataset.apply(tf.data.experimental.prefetch_to_device(device)) # dà errore su GPU

                # generate the dataset for evaluation
                val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
                val_dataset = val_dataset.batch(self.batchsize)
                #val_dataset = val_dataset.repeat()
                val_dataset = val_dataset.prefetch(4)  # da usare se prefetch_to_device non funziona
                # val_dataset = val_dataset.apply(tf.data.experimental.prefetch_to_device(device)) # dà errore su GPU

                num_train_batches = x_train.shape[0] // self.batchsize + 1

                global_step = tf.compat.v1.train.get_or_create_global_step()
                lr = tf.compat.v1.train.cosine_decay(self.lr, global_step, decay_steps=num_train_batches * self.epochs, alpha=0.1)

                # TODO: original PNAS don't mention Adam, only cosine decay
                # construct the optimizer and saver of the child model
                optimizer = tf.compat.v1.train.AdamOptimizer(lr)
                saver = tf.train.Checkpoint(model=model, optimizer=optimizer, global_step=global_step)

                best_val_acc = 0.0
                timer = 0  # inizialize timer to evaluate training time

                for epoch in range(self.epochs):
                    # train child model
                    with tqdm(iterable=enumerate(train_dataset),
                              desc=f'Train Epoch ({epoch + 1} / {self.epochs}): ',
                              unit='batch',
                              total=num_train_batches) as pbar:

                        for _, (x, y) in pbar:
                            # get gradients
                            with tf.GradientTape() as tape:
                                # get training starting time
                                start = time.clock()
                                preds = model(x, training=True)
                                loss = tf.keras.losses.categorical_crossentropy(y, preds)

                            grad = tape.gradient(loss, model.variables)
                            grad_vars = zip(grad, model.variables)

                            # update weights of the child models
                            optimizer.apply_gradients(grad_vars, global_step)

                            # get training ending time
                            stop = time.clock()

                            # evaluate training time
                            timer = timer + (stop - start)

                    # evaluate child model
                    acc = tf.metrics.CategoricalAccuracy()
                    for _, (x, y) in enumerate(val_dataset):
                        preds = model(x, training=False)
                        acc(y, preds)

                    acc = acc.result().numpy()

                    # add forward pass and accuracy to Tensorboard
                    with self.summary_writer.as_default():
                        tf.summary.scalar("training_time", timer, description="children", step=epoch+1)
                        summary_acc = acc if acc > best_val_acc else best_val_acc
                        tf.summary.scalar("child_accuracy", summary_acc, description="children", step=epoch+1)

                    self._logger.info("\tEpoch %d: Training time = %0.6f", epoch + 1, timer)
                    self._logger.info("\tEpoch %d: Val accuracy = %0.6f", epoch + 1, acc)

                    # if acc improved, save the weights
                    if acc > best_val_acc:
                        self._logger.info("\tVal accuracy improved from %0.6f to %0.6f. Saving weights!", best_val_acc, acc)

                        best_val_acc = acc
                        saver.save('temp_weights/temp_network')

                # test_writer.close()

                # load best weights of the child model
                path = tf.train.latest_checkpoint('temp_weights/')
                saver.restore(path)

                # display the structure of the child model
                if display_model_summary:
                    model.summary(line_length=140, print_fn=self._logger.info)
                    # plot_model(model, to_file='%s/model_plot.png' % self.logdir, show_shapes=True, show_layer_names=True)

                # evaluate the best weights of the child model
                acc = tf.metrics.CategoricalAccuracy()

                for j, (x, y) in enumerate(val_dataset):
                    preds = model(x, training=False)
                    acc(y, preds)

                acc = acc.result().numpy()
                acc_list.append(acc)

            # compute the reward (validation accuracy)
            reward = acc

            self._logger.info("Manager: Accuracy = %0.6f", reward)

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
        del optimizer
        del global_step

        return [reward, timer]
