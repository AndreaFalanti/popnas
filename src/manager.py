import csv
import gc
import logging
import os
import shutil
from statistics import mean
from typing import Tuple

import absl.logging
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend
import tf2onnx
from matplotlib import pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model

import log_service
from model import ModelGenerator
from utils.dataset_utils import generate_tensorflow_datasets, get_data_augmentation_model, generate_balanced_weights_for_classes
from utils.func_utils import cell_spec_to_str
from utils.graph_generator import GraphGenerator
from utils.nn_utils import get_best_val_accuracy_per_output, get_model_flops, get_optimized_steps_per_execution, save_keras_model_to_onnx, \
    TrainingResults
from utils.timing_callback import TimingCallback, InferenceTimingCallback

absl.logging.set_verbosity(absl.logging.ERROR)  # disable strange useless warning in model saving, that is also present in TF tutorial...

# disable Tensorflow info and warning messages (Warning are not on important things, they were investigated. Still, enable them
# when performing changes to see if there are new potential warnings that can affect negatively the algorithm).
tf.get_logger().setLevel(logging.ERROR)

# disable tf2onnx conversion messages
tf2onnx.logging.set_level(logging.WARN)

AUTOTUNE = tf.data.AUTOTUNE


class NetworkManager:
    '''
    Helper class to manage the generation of subnetwork training given a dataset
    '''

    def __init__(self, dataset_config: dict, cnn_config: dict, arc_config: dict, train_strategy: tf.distribute.Strategy,
                 save_network_weights: bool, save_as_onnx: bool):
        '''
        Manager which is tasked with creating subnetworks, training them on a dataset, and retrieving
        rewards in the term of accuracy, which is passed to the controller RNN.
        It also preprocess the dataset, based on the run configuration.
        '''
        self._logger = log_service.get_logger(__name__)

        self.dataset_folds_count = dataset_config['folds']
        self.dataset_classes_count = dataset_config['classes_count']
        self.balance_class_losses = dataset_config['balance_class_losses']
        self.augment_on_gpu = dataset_config['data_augmentation']['enabled'] and dataset_config['data_augmentation']['perform_on_gpu']

        self.epochs = cnn_config['epochs']
        self.train_strategy = train_strategy

        self.num_child = 0  # SUMMARY
        self.best_reward = 0.0

        # setup dataset. Batches variables are used for displaying progress during training
        self.dataset_folds, ds_classes, image_shape, self.train_batches, self.validation_batches = \
            generate_tensorflow_datasets(dataset_config, self._logger)
        self.dataset_classes_count = ds_classes or self.dataset_classes_count   # Javascript || operator
        self.balanced_class_weights = [generate_balanced_weights_for_classes(train_ds) for train_ds, _ in self.dataset_folds] \
            if self.balance_class_losses else None

        self.model_gen = ModelGenerator(cnn_config, arc_config, self.train_batches,
                                        output_classes=self.dataset_classes_count, image_shape=image_shape,
                                        data_augmentation_model=get_data_augmentation_model() if self.augment_on_gpu else None,
                                        save_weights=save_network_weights)

        # TODO: if not needed here, just generate it in train to pass it in controller
        self.graph_gen = GraphGenerator(cnn_config, arc_config, image_shape, self.dataset_classes_count)

        self.multi_output_model = arc_config['multi_output']
        self.multi_output_csv_headers = [f'c{i}_accuracy' for i in range(self.model_gen.total_cells)] + ['cell_spec']

        self.save_onnx = save_as_onnx

        # take 10 batches of size provided in config, used to check the inference time on that batch size with
        self.inference_batch_size = dataset_config['inference_batch_size']
        self.inference_batch = self.dataset_folds[0][0].unbatch().take(self.inference_batch_size * 10).batch(self.inference_batch_size)

        # DEBUG ONLY
        # self.__test_data_augmentation(self.dataset_folds[0][0])

    def __test_data_augmentation(self, ds: tf.data.Dataset):
        '''
        Function helpful for debugging data augmentation and making sure it's working properly.
        DON'T USE IT IN ACTUAL RUNS.
        Args:
            ds: any TF dataset where data augmentation is applied
        '''
        # switch to an interactive matplotlib backend
        plt.switch_backend('TkAgg')

        data_augmentation_model = get_data_augmentation_model()

        # get a batch
        images, labels = next(iter(ds))

        # display 9 transformation of the first 3 images of the first training batch
        for j in range(3):
            image = images[j]
            plt.imshow(image)
            plt.show()

            for i in range(9):
                augmented_image = data_augmentation_model(image)
                _ = plt.subplot(3, 3, i + 1)
                plt.imshow(augmented_image)
                plt.axis('off')

            plt.show()
        self._logger.debug('Data augmentation debug shown')

    def __write_partitions_file(self, partition_dict: dict, save_dir: str):
        lines = [f'{key}: {value:,} bytes' for key, value in partition_dict.items()]

        with open(save_dir, 'w') as f:
            # writelines function usually add \n automatically, but not in python...
            f.writelines(line + '\n' for line in lines)

    def __write_multi_output_file(self, cell_spec: list, outputs_dict: dict):
        # add cell spec to dictionary and write it into the csv
        outputs_dict['cell_spec'] = cell_spec_to_str(cell_spec)

        with open(log_service.build_path('csv', 'multi_output.csv'), mode='a+', newline='') as f:
            # append mode, so if file handler is in position 0 it means is empty. In this case write the headers too
            if f.tell() == 0:
                writer = csv.writer(f)
                writer.writerow(self.multi_output_csv_headers)

            writer = csv.DictWriter(f, self.multi_output_csv_headers)
            writer.writerow(outputs_dict)

    def __compile_model(self, cell_spec: 'list[tuple]', tb_logdir: str) -> Tuple[Model, list, dict]:
        '''
        Generate and compile a Keras model, with cell structure defined by actions provided.

        Args:
            cell_spec (list): [description]
            tb_logdir (str): path for tensorboard logging

        Returns:
            (tf.keras.Model, list[tf.keras.callbacks.Callback], dict): model and callbacks to use while training
        '''

        with self.train_strategy.scope():
            model, partition_dict, _ = self.model_gen.build_model(cell_spec)

            loss, loss_weights, optimizer, metrics = self.model_gen.define_training_hyperparams_and_metrics()
            execution_steps = get_optimized_steps_per_execution(self.train_strategy)
            model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights, metrics=metrics, steps_per_execution=execution_steps)

        # for debugging keras layers, otherwise leave this commented since it will destroy performance
        # model.run_eagerly = True

        if self.save_onnx:
            save_keras_model_to_onnx(model, os.path.join(tb_logdir, 'model.onnx'))
            self._logger.info('Equivalent ONNX model serialized successfully and saved to file')

        return model, self.model_gen.define_callbacks(tb_logdir), partition_dict

    def bootstrap_dataset_lazy_initialization(self):
        '''
        Train the empty cell model for a single epoch, just to load, process and cache the dataset, so that the first model trained in the session
        is not affected by a time estimation bias (which can be very large for datasets generated from image folders).
        '''
        self._logger.info('Performing an epoch of training to lazy load and cache the dataset')
        fake_tb_logdir = log_service.build_path('tensorboard_cnn', f'Bx')
        model, callbacks, partition_dict = self.__compile_model([], fake_tb_logdir)

        for i, (train_ds, val_ds) in enumerate(self.dataset_folds):
            if self.dataset_folds_count > 1:
                self._logger.info("Training on dataset #%d / #%d", i + 1, self.dataset_folds_count)

            model.fit(x=train_ds,
                      epochs=1,
                      steps_per_epoch=self.train_batches,
                      validation_data=val_ds,
                      validation_steps=self.validation_batches,
                      class_weight=self.balanced_class_weights[i] if self.balance_class_losses else None)

        # remove useless directory
        shutil.rmtree(fake_tb_logdir, ignore_errors=True)
        self._logger.info('Dummy training complete, training time should now be unaffected by dataset lazy initialization')


    def perform_proxy_training(self, cell_spec: 'list[tuple]', save_best_model: bool = False):
        '''
        Generate a neural network from the cell specification and trains it for a short amount of epochs to get an estimate
        of its quality. Other relevant metrics of the NN architecture, like the params and flops, are returned together with the training results.

        Args:
            cell_spec (list[tuple]): plain cell specification. Used to build the CNN.
            save_best_model (bool, optional): [description]. Defaults to False.

        Returns:
            (TrainingResults): (reward, timer, total_params, flops, inference_time) of trained network
        '''

        # TODO: legacy function, don't know why it was called. Doesn't seem to harm the execution, if you feel brave
        #  try to remove it and check if something is wrong.
        tf.keras.backend.reset_uids()

        # create children folder on Tensorboard
        self.num_child = self.num_child + 1
        # grouped for block count and enumerated progressively
        tb_logdir = log_service.build_path('tensorboard_cnn', f'B{len(cell_spec)}', str(self.num_child))
        os.makedirs(tb_logdir, exist_ok=True)

        # store training results for each fold
        times = np.empty(shape=self.dataset_folds_count, dtype=np.float64)
        accuracies = np.empty(shape=self.dataset_folds_count, dtype=np.float64)

        for i, (train_ds, val_ds) in enumerate(self.dataset_folds):
            if self.dataset_folds_count > 1:
                self._logger.info("Training on dataset #%d / #%d", i + 1, self.dataset_folds_count)

            # generate a CNN model given the cell specification
            # TODO: instead of rebuilding the model it should be better to just reset the weights and the optimizer
            model, callbacks, partition_dict = self.__compile_model(cell_spec, tb_logdir)
            # add callback to register as accurate as possible the training time
            time_cb = TimingCallback()
            callbacks.append(time_cb)

            hist = model.fit(x=train_ds,
                             epochs=self.epochs,
                             steps_per_epoch=self.train_batches,
                             validation_data=val_ds,
                             validation_steps=self.validation_batches,
                             callbacks=callbacks,
                             class_weight=self.balanced_class_weights[i] if self.balance_class_losses else None)

            times[i] = sum(time_cb.logs)
            # compute the reward (best validation accuracy)
            if self.multi_output_model and len(cell_spec) > 0:
                multi_output_accuracies = get_best_val_accuracy_per_output(hist)

                # use as val accuracy metric the best one among all softmax layers
                accuracies[i] = max(multi_output_accuracies.values())
                self.__write_multi_output_file(cell_spec, multi_output_accuracies)
            else:
                accuracies[i] = max(hist.history['val_accuracy'])

        training_time = times.mean()
        accuracy = accuracies.mean()
        total_params = model.count_params()
        # TODO: bugged on Google Cloud TPU VMs, since they seems to lack CPU:0 device (hidden by environment or strategy?).
        #  Set to 0 on TPUs for now since it is only retrieved for additional analysis, take care if using FLOPs in algorithm.
        flops = get_model_flops(model, os.path.join(tb_logdir, 'flops_log.txt')) if not isinstance(self.train_strategy, tf.distribute.TPUStrategy) \
            else 0

        # compute inference time
        inference_time_cb = InferenceTimingCallback()
        model.predict(self.inference_batch, callbacks=[inference_time_cb])
        # discard first batch time since it is very noisy due to initialization of the process
        inference_time = mean(inference_time_cb.logs[1:])

        # write model summary to file
        with open(os.path.join(tb_logdir, 'summary.txt'), 'w') as f:
            # str casting is required since inputs are int
            f.write('Cell specification: ' + ';'.join(map(str, cell_spec)) + '\n\n')
            model.summary(line_length=150, print_fn=lambda x: f.write(x + '\n'))
            f.write(f'\nFLOPS: {flops:,}')

        # write partitions file
        self.__write_partitions_file(partition_dict, os.path.join(tb_logdir, 'partitions.txt'))

        # save also an overview diagram of the network
        plot_model(model, to_file=os.path.join(tb_logdir, 'model.pdf'), show_shapes=True, show_layer_names=True)

        # if algorithm is training the last models batch (B = value provided in command line)
        # save the best model in a folder, so that can be trained from scratch later on
        if save_best_model and accuracy > self.best_reward:
            self.best_reward = accuracy
            # last model should be automatically overwritten, leaving only one model
            self._logger.info('Saving model...')
            model.save(log_service.build_path('best_model', 'saved_model.h5'), save_format='h5')
            save_keras_model_to_onnx(model, log_service.build_path('best_model', 'saved_model.onnx'))
            self._logger.info('Model saved successfully')

        # clean up resources and GPU memory (TODO: actually solving TPU_VM problem or not?)
        tensorflow.keras.backend.clear_session()
        gc.collect()
        del model

        return TrainingResults(cell_spec, accuracy, training_time, total_params, flops, inference_time)
