import csv
import importlib.util
import logging
import os
import re
from typing import Tuple

import numpy as np
import tensorflow as tf
import tf2onnx
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model, datasets, layers, Sequential
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph

import log_service
from model import ModelGenerator
from utils.func_utils import cell_spec_to_str
from utils.timing_callback import TimingCallback

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)  # disable strange useless warning in model saving, that is also present in TF tutorial...

# TODO: not working properly, thx tensorflow for the "Callback method `on_train_batch_end` is slow compared to the batch time" at each training...
# disable Tensorflow info and warning messages (Warning are not on important things, they were investigated. Still, enable them
# when performing changes to see if there are new potential warnings that can affect negatively the algorithm.
tf.get_logger().setLevel(logging.ERROR)

# disable tf2onnx conversion messages
tf2onnx.logging.set_level(logging.WARN)

AUTOTUNE = tf.data.AUTOTUNE


# tentative way of reproducing PNAS transformation
# TODO: usable in image generator as preprocessing function, unfortunately slow and seems not good for learning too...
def upsample_and_random_crop(image_tensor):
    image_tensor = tf.image.resize(image_tensor, [40, 40])
    image_tensor = tf.image.random_crop(image_tensor, [32, 32, 3])
    # normalize in [-1, 1] domain
    return preprocess_input(image_tensor, mode='tf')


class NetworkManager:
    '''
    Helper class to manage the generation of subnetwork training given a dataset
    '''

    def __init__(self, dataset_config: dict, cnn_config: dict, arc_config: dict):
        '''
        Manager which is tasked with creating subnetworks, training them on a dataset, and retrieving
        rewards in the term of accuracy, which is passed to the controller RNN.
        It also preprocess the dataset, based on the run configuration.
        '''
        self._logger = log_service.get_logger(__name__)

        self.dataset_name = dataset_config['name']
        self.dataset_path = dataset_config['path']
        self.dataset_folds_count = dataset_config['folds']
        self.samples_limit = dataset_config['samples']
        self.dataset_classes_count = dataset_config['classes_count']

        data_augmentation_config = dataset_config['data_augmentation']
        self.use_data_augmentation = data_augmentation_config['enabled']
        self.augment_on_gpu = data_augmentation_config['perform_on_gpu']

        self.dataset_folds = []  # type: list[tuple[tf.data.Dataset, tf.data.Dataset]]

        self.epochs = cnn_config['epochs']
        self.batch_size = cnn_config['batch_size']

        self.num_child = 0  # SUMMARY
        self.best_reward = 0.0

        # Keras model that can be used in both CPU or GPU for data augmentation
        if self.use_data_augmentation:
            # follow similar augmentation techniques used in other papers, which usually are:
            # - horizontal flip
            # - 4px translate on both height and width [fill=reflect] (sometimes upscale to 40x40, with random crop to original 32x32)
            # - whitening (not always used)
            self.data_augmentation = Sequential([
                layers.experimental.preprocessing.RandomFlip('horizontal'),
                # layers.experimental.preprocessing.RandomRotation(20/360),   # 20 degrees range
                # layers.experimental.preprocessing.RandomZoom(height_factor=0.1, width_factor=0.1),
                layers.experimental.preprocessing.RandomTranslation(height_factor=0.125, width_factor=0.125)
            ], name='data_augmentation')
        else:
            self.data_augmentation = None

        # set in dataset initialization, used for displaying progress during training
        self.train_batches = 0
        self.validation_batches = 0

        (x_train_init, y_train_init), _ = self.__load_dataset()
        self.__prepare_datasets(x_train_init, y_train_init)
        self._logger.info('Dataset folds built successfully')

        self.model_gen = ModelGenerator(cnn_config, arc_config, self.train_batches, output_classes=self.dataset_classes_count,
                                        data_augmentation_model=self.data_augmentation if self.augment_on_gpu else None)

        self.multi_output_model = arc_config['multi_output']
        self.multi_output_csv_headers = [f'c{i}_accuracy' for i in range(self.model_gen.total_cells)] + ['cell_spec']

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

        # get a batch
        images, labels = next(iter(ds))

        # display 9 transformation of the first 3 images of the first training batch
        for j in range(3):
            image = images[j]
            plt.imshow(image)
            plt.show()

            for i in range(9):
                augmented_image = self.data_augmentation(image)
                _ = plt.subplot(3, 3, i + 1)
                plt.imshow(augmented_image)
                plt.axis('off')

            plt.show()
        self._logger.debug('Data augmentation debug shown')

    def __load_dataset(self):
        self._logger.info('Loading dataset...')
        # for known names, load dataset from Keras and set the correct number of classes, in case there is a mismatch in the json
        if self.dataset_name == 'cifar10':
            (x_train_init, y_train_init), (x_test_init, y_test_init) = datasets.cifar10.load_data()
            self.dataset_classes_count = 10
        elif self.dataset_name == 'cifar100':
            (x_train_init, y_train_init), (x_test_init, y_test_init) = datasets.cifar100.load_data()
            self.dataset_classes_count = 100
        # TODO: untested legacy code, not sure this is working
        # if dataset name is not recognized, try to import it from path
        else:
            spec = importlib.util.spec_from_file_location(self.dataset_path)
            dataset = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(dataset)
            (x_train_init, y_train_init), (x_test_init, y_test_init) = dataset.load_data()

        return (x_train_init, y_train_init), (x_test_init, y_test_init)

    def __prepare_datasets(self, x_train_init, y_train_init):
        """Build a validation set from training set and do some preprocessing

        Args:
            x_train_init (ndarray): x training
            y_train_init (ndarray): y training

        Returns:
            list[tuple(tf.data.Dataset, tf.data.Dataset)]: list with pairs of training and validation datasets, for each fold
        """
        if self.samples_limit is not None:
            x_train_init = x_train_init[:self.samples_limit]
            y_train_init = y_train_init[:self.samples_limit]

        # normalize image RGB values into [0, 1] domain
        x_train_init = x_train_init.astype('float32') / 255.
        y_train_init = to_categorical(y_train_init, self.dataset_classes_count)

        # TODO: is it ok to generate the splits by shuffling randomly?
        for i in range(self.dataset_folds_count):
            self._logger.info('Preprocessing and building dataset fold #%d...', i + 1)

            # create a validation set for evaluation of the child models
            x_train, x_validation, y_train, y_validation = train_test_split(x_train_init, y_train_init, test_size=0.1, stratify=y_train_init)

            train_ds, val_ds, train_batches, val_batches = self.__build_tf_datasets((x_train, y_train, x_validation, y_validation))
            self.train_batches = train_batches
            self.validation_batches = val_batches

            self.dataset_folds.append((train_ds, val_ds))

    def __build_tf_datasets(self, samples_fold: 'tuple(list, list, list, list)'):
        '''
        Build the training and validation datasets to be used in model.fit().

        Args:
            samples_fold: dataset index
            use_data_augmentation (bool): [description]

        Returns:
            [type]: [description]
        '''

        x_train, y_train, x_val, y_val = samples_fold

        # TODO: legacy, ImageDataGenerator is very slow when multiple transformations are performed, instead are handled nicely in both CPU
        #  and GPU with the Keras model. On horizontal flip only it is faster, but terribly slower on the rest.
        #  Left here for eventual further testing.
        # if self.use_data_augmentation:
        #     train_datagen = ImageDataGenerator(
        #         horizontal_flip=True,
        #         width_shift_range=4,
        #         height_shift_range=4
        #     )
        # else:
        #     train_datagen = ImageDataGenerator()
        #
        # train_datagen.fit(x_train)
        # validation_datagen = ImageDataGenerator()
        #
        # # TODO: generalize for other datasets
        # cifar10_signature = (tf.TensorSpec(shape=(None, 32, 32, 3), dtype=tf.float32),
        #                      tf.TensorSpec(shape=(None, self.dataset_classes_count), dtype=tf.float32))
        # train_dataset = tf.data.Dataset.from_generator(train_datagen.flow, args=(x_train, y_train, self.batch_size),
        #                                                output_signature=cifar10_signature).prefetch(AUTOTUNE)
        # validation_dataset = tf.data.Dataset.from_generator(validation_datagen.flow, args=(x_val, y_val, self.batch_size),
        #                                                     output_signature=cifar10_signature).prefetch(AUTOTUNE)

        # create a batched dataset, cached in memory for better performance
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(self.batch_size).cache()
        validation_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(self.batch_size).cache().prefetch(AUTOTUNE)

        # if data augmentation is performed on CPU, map it before prefetch, otherwise just prefetch
        train_dataset = train_dataset.prefetch(AUTOTUNE) if self.augment_on_gpu\
            else train_dataset.map(lambda x, y: (self.data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

        train_batches = np.ceil(len(x_train) / self.batch_size)
        val_batches = np.ceil(len(x_val) / self.batch_size)

        return train_dataset, validation_dataset, int(train_batches), int(val_batches)

    # See: https://github.com/tensorflow/tensorflow/issues/32809#issuecomment-768977280
    # See also: https://stackoverflow.com/questions/49525776/how-to-calculate-a-mobilenet-flops-in-keras
    def get_model_flops(self, model, write_path=None):
        '''
        Get total flops of current compiled model.

        Returns:
            (int): number of FLOPS
        '''
        # temporarily disable warnings, a lot of them are print and they seems irrelevant
        # TODO: investigate if something is actually wrong
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

        concrete = tf.function(lambda inputs: model(inputs))
        concrete_func = concrete.get_concrete_function([tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in model.inputs])

        frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)

        with tf.Graph().as_default() as graph:
            tf.graph_util.import_graph_def(graph_def, name='')
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            if write_path is not None:
                opts['output'] = 'file:outfile={}'.format(write_path)  # redirect output
            else:
                opts['output'] = 'none'

            flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)

        # enable again warnings
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)

        return flops.total_float_ops

    def __write_partitions_file(self, partition_dict: dict, save_dir: str):
        lines = [f'{key}: {value:,} bytes' for key, value in partition_dict.items()]

        with open(save_dir, 'w') as f:
            # GG python devs for this crap, a writelines function that works like a write, not adding \n automatically...
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

        model, partition_dict = self.model_gen.build_model(cell_spec)

        loss, loss_weights, optimizer, metrics = self.model_gen.define_training_hyperparams_and_metrics()
        model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights, metrics=metrics)

        # for debugging keras layers, otherwise leave this commented since it will destroy performance
        # model.run_eagerly = True

        onnx_model = tf2onnx.convert.from_keras(model, opset=10)
        with open(os.path.join(tb_logdir, 'model.onnx'), 'wb') as f:
            f.write(onnx_model[0].SerializeToString())
        self._logger.info('Equivalent ONNX model serialized successfully and saved to file')

        return model, self.model_gen.define_callbacks(tb_logdir), partition_dict

    def get_rewards(self, cell_spec: 'list[tuple]', save_best_model: bool = False):
        '''
        Creates a CNN given the actions predicted by the controller RNN,
        trains it on the provided dataset, and then returns a reward.

        Args:
            cell_spec (list[tuple]): plain cell specification. Used to build the CNN.
            save_best_model (bool, optional): [description]. Defaults to False.

        Returns:
            (tuple): (reward, timer, total_params, flops) of trained network
        '''

        # TODO: don't know why it was called. Try to remove it and check if something is wrong.
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
                             batch_size=self.batch_size,
                             steps_per_epoch=self.train_batches,
                             validation_data=val_ds,
                             validation_steps=self.validation_batches,
                             callbacks=callbacks)

            times[i] = sum(time_cb.logs)
            # compute the reward (best validation accuracy)
            if self.multi_output_model and len(cell_spec) > 0:
                # use as val accuracy metric only the one of the softmax placed after the last cell
                r = re.compile(r'val_Softmax_c(\d+)_accuracy')
                output_indexes = [int(match.group(1)) for match in map(r.match, hist.history.keys()) if match]
                max_index = max(output_indexes)
                accuracies[i] = max(hist.history[f'val_Softmax_c{max_index}_accuracy'])

                # save best accuracy reached for each output
                multi_output_accuracies = {}
                for output_index in output_indexes:
                    multi_output_accuracies[f'c{output_index}_accuracy'] = max(hist.history[f'val_Softmax_c{output_index}_accuracy'])

                self.__write_multi_output_file(cell_spec, multi_output_accuracies)
            else:
                accuracies[i] = max(hist.history['val_accuracy'])

        training_time = times.mean()
        reward = accuracies.mean()
        total_params = model.count_params()
        flops = self.get_model_flops(model, os.path.join(tb_logdir, 'flops_log.txt'))

        # write model summary to file
        with open(os.path.join(tb_logdir, 'summary.txt'), 'w') as f:
            # str casting is required since inputs are int
            f.write('Model actions: ' + ','.join(map(lambda el: str(el), cell_spec)) + '\n\n')
            model.summary(line_length=150, print_fn=lambda x: f.write(x + '\n'))
            f.write(f'\nFLOPS: {flops:,}')

        # write partitions file
        self.__write_partitions_file(partition_dict, os.path.join(tb_logdir, 'partitions.txt'))

        # save also an overview diagram of the network
        plot_model(model, to_file=os.path.join(tb_logdir, 'model.png'), show_shapes=True, show_layer_names=True)

        # if algorithm is training the last models batch (B = value provided in command line)
        # save the best model in a folder, so that can be trained from scratch later on
        if save_best_model and reward > self.best_reward:
            self.best_reward = reward
            # last model should be automatically overwritten, leaving only one model
            self._logger.info('Saving model...')
            model.save(log_service.build_path('best_model', 'saved_model.h5'), save_format='h5')
            self._logger.info('Model saved successfully')

        # clean up resources and GPU memory
        del model

        return reward, training_time, total_params, flops
