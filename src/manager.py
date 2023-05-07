import os
import shutil
from statistics import mean

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import History, Callback
from tensorflow.keras.utils import plot_model

import file_writer as fw
import log_service
from dataset.generators.factory import dataset_generator_factory
from dataset.utils import generate_balanced_weights_for_classes
from models.custom_callbacks import InferenceTimingCallback, TrainingTimeCallback
from models.generators.factory import model_generator_factory
from models.results.base import write_multi_output_results_to_csv
from search_space_units import CellSpecification
from utils.config_dataclasses import *
from utils.nn_utils import get_model_flops, get_optimized_steps_per_execution, save_keras_model_to_onnx, perform_global_memory_clear
from utils.rstr import rstr

AUTOTUNE = tf.data.AUTOTUNE


class NetworkManager:
    '''
    Helper class to manage the generation of subnetwork training given a dataset
    '''

    def __init__(self, dataset_config: DatasetConfig, train_config: TrainingHyperparametersConfig, arc_config: ArchitectureHyperparametersConfig,
                 others_config: OthersConfig, score_objective: str, train_strategy: tf.distribute.Strategy):
        '''
        Manager which is tasked with creating subnetworks, training them on a dataset, and retrieving
        rewards in the term of accuracy, which is passed to the controller RNN.
        It also initializes the dataset pipeline, based on the run configuration.
        '''
        self._logger = log_service.get_logger(__name__)

        self.dataset_folds_count = dataset_config.folds
        self.dataset_classes_count = dataset_config.classes_count

        self.epochs = train_config.epochs
        self.train_strategy = train_strategy
        self.execution_steps = get_optimized_steps_per_execution(self.train_strategy)

        # integer id assigned to each network trained, name of the folder containing the training outputs (see "sampled_models" folder)
        self.current_network_id = 0
        # fix for correcting the network id automatically when restoring a previous run
        train_csv_path = log_service.build_path('csv', 'training_results.csv')
        if os.path.isfile(train_csv_path):
            with open(train_csv_path, 'r') as f:
                self.current_network_id = len(f.readlines()) - 1

        self.best_score = 0.0
        self.score_objective = score_objective

        # setup dataset; batches variables are used for displaying progress during training
        dataset_generator = dataset_generator_factory(dataset_config, others_config)
        self.dataset_folds, ds_classes, input_shape, self.train_batches, self.validation_batches, preprocessing_model \
            = dataset_generator.generate_train_val_datasets()
        self.dataset_classes_count = ds_classes or self.dataset_classes_count  # Javascript || operator
        # setting weights to None is the default option to not regularize classes
        self.balanced_class_weights = [generate_balanced_weights_for_classes(train_ds) for train_ds, _ in self.dataset_folds] \
            if dataset_config.balance_class_losses else [None] * len(self.dataset_folds)

        self.model_gen = model_generator_factory(dataset_config, train_config, arc_config, self.train_batches,
                                                 output_classes_count=self.dataset_classes_count, input_shape=input_shape,
                                                 preprocessing_model=preprocessing_model,
                                                 save_weights=others_config.save_children_weights)
        self.TrainingResults = self.model_gen.get_results_processor_class()

        self.multi_output_model = arc_config.multi_output
        self.multi_output_csv_headers = [f'c{i}_{m.name}' for m in self.TrainingResults.keras_metrics_considered()
                                         for i in range(self.model_gen.get_maximum_cells())] + ['cell_spec']

        self.save_all_models = others_config.save_children_models
        self.XLA_compile = others_config.enable_XLA_compilation

        # take N batches of size provided in config, used to test the inference time.
        # when using multiple steps per execution, multiply the number of batches by the steps executed.
        inference_trials = 13
        self.inference_batch_size = dataset_config.inference_batch_size
        self.inference_batches_count = inference_trials * self.execution_steps
        # use training split since it always has a fixed size, making it more probable to have consistent inference measurements
        self.inference_batch = self.dataset_folds[0].train.unbatch() \
            .take(self.inference_batch_size * self.inference_batches_count).batch(self.inference_batch_size)

        # DEBUG ONLY
        # test_data_augmentation(self.dataset_folds[0][0])

    def __compile_model(self, cell_spec: CellSpecification, model_logdir: str) -> 'tuple[Model, list[Callback]]':
        '''
        Generate and compile a Keras model, with cell structure defined by actions provided.

        Args:
            cell_spec: cell specification
            model_logdir: path for storing logs and data about the trained model

        Returns:
            model and callbacks to use while training
        '''

        with self.train_strategy.scope():
            model, _ = self.model_gen.build_model(cell_spec)

            loss, loss_weights, optimizer, metrics = self.model_gen.define_training_hyperparams_and_metrics()
            model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights, metrics=metrics,
                          steps_per_execution=self.execution_steps, jit_compile=self.XLA_compile)

        # for debugging keras layers, otherwise leave this commented since it will destroy performance
        # model.run_eagerly = True

        return model, self.model_gen.define_callbacks(model_logdir, self.score_objective)

    def _save_model(self, model: Model, save_path: str):
        ''' Save the model in ONNX format. Any previous model in the same path is automatically overwritten, leaving only the last model saved. '''
        self._logger.info('Saving model at path: %s', save_path)
        model.save(os.path.join(save_path, 'tf_model'))
        self._logger.info('TF model saved successfully')
        save_keras_model_to_onnx(model, os.path.join(save_path, 'model.onnx'))
        self._logger.info('Equivalent ONNX model serialized successfully and saved to file')

    def bootstrap_dataset_lazy_initialization(self):
        '''
        Train the empty cell model for a single epoch, just to load, process and cache the dataset, so that the first model trained in the session
        is not affected by a time estimation bias (which can be very large for datasets generated from image folders).
        '''
        self._logger.info('Performing an epoch of training to lazy load and cache the dataset')
        fake_tb_logdir = log_service.build_path('sampled_models', f'Bx')
        model, callbacks = self.__compile_model(CellSpecification(), fake_tb_logdir)

        for i, (train_ds, val_ds) in enumerate(self.dataset_folds):
            if self.dataset_folds_count > 1:
                self._logger.info("Training on dataset #%d / #%d", i + 1, self.dataset_folds_count)

            model.fit(x=train_ds,
                      epochs=1,
                      steps_per_epoch=self.train_batches,
                      validation_data=val_ds,
                      validation_steps=self.validation_batches,
                      class_weight=self.balanced_class_weights[i])

        # remove useless directory
        shutil.rmtree(fake_tb_logdir, ignore_errors=True)
        self._logger.info('Dummy training complete, training time should now be unaffected by dataset lazy initialization')

    def perform_proxy_training(self, cell_spec: CellSpecification, save_best_model: bool = False):
        '''
        Generate a neural network from the cell specification and trains it for a short number of epochs to get an estimate
        of its quality. Other relevant metrics of the NN architecture, like the params and flops, are returned together with the training results.

        Args:
            cell_spec: plain cell specification. Used to build the CNN.
            save_best_model: if best model of b=B should be saved as ONNX. Defaults to False.

        Returns:
            results and characteristics of trained network
        '''

        # reset Keras layer naming counters
        tf.keras.backend.reset_uids()

        # create children folder for Tensorboard logs, grouped for block count and enumerated progressively
        self.current_network_id = self.current_network_id + 1
        model_logdir = log_service.build_path('sampled_models', f'B{len(cell_spec)}', str(self.current_network_id))
        os.makedirs(model_logdir, exist_ok=True)

        model, histories, times = self.train_model(cell_spec, model_logdir)

        training_time = times.mean()
        total_params = model.count_params()
        # avoid computing FLOPs when the input size is not completely defined (networks supporting different sample sizes can have dims = None)
        # TODO: bugged on Google Cloud TPU VMs, since they seems to lack CPU:0 device (hidden by environment or strategy?).
        #  Set to 0 on TPUs for now since it is only retrieved for additional analysis, take care if using FLOPs in algorithm.
        if not any(dim is None for dim in self.model_gen.input_shape) and not isinstance(self.train_strategy, tf.distribute.TPUStrategy):
            flops = get_model_flops(model, os.path.join(model_logdir, 'flops_log.txt'))
        else:
            flops = 0

        inference_time = self.check_inference_speed(model)

        # empty cell has a single output even if the multi-output flag is set
        is_multi_output = self.multi_output_model and not cell_spec.is_empty_cell()
        training_res = self.TrainingResults.from_training_histories(cell_spec, training_time, inference_time, total_params, flops,
                                                                    histories, is_multi_output)

        # save ONNX and TF models for each child, if the related flag is enabled in the JSON config
        if self.save_all_models:
            self._save_model(model, save_path=model_logdir)

        # write additional model files
        fw.write_model_summary_file(cell_spec, flops, model, os.path.join(model_logdir, 'summary.txt'))
        partition_dict = self.model_gen.compute_network_partitions(cell_spec, tensor_dtype=tf.float32)
        fw.write_partitions_file(partition_dict, os.path.join(model_logdir, 'partitions.txt'))
        plot_model(model, to_file=os.path.join(model_logdir, 'model.pdf'), show_shapes=True, show_layer_names=True)
        if is_multi_output:
            multi_output_csv_path = log_service.build_path('csv', 'multi_output.csv')
            write_multi_output_results_to_csv(multi_output_csv_path, cell_spec, histories,
                                              self.TrainingResults.keras_metrics_considered(), self.multi_output_csv_headers)

        # if the algorithm is training the last batch of models (B = value provided in command line),
        # save the best model in a folder, so that can be trained from scratch later on.
        score = getattr(training_res, self.score_objective)
        # avoid saving ONNX on TPU (CPU:0 device not found, same as flops)
        if save_best_model and score > self.best_score and not isinstance(self.train_strategy, tf.distribute.TPUStrategy):
            self.best_score = score
            self._save_model(model, save_path=log_service.build_path('best_model'))

        perform_global_memory_clear()

        return training_res

    def check_inference_speed(self, model: Model):
        ''' Compute the given model inference time by executing a short prediction session. '''
        self._logger.info('Testing inference speed...')
        inference_time_cb = InferenceTimingCallback()
        model.predict(self.inference_batch, steps=self.inference_batches_count, callbacks=[inference_time_cb])

        # discard the first 3 batch measurements, since they can be pretty noisy due to initialization of the process (used as warmup of the model)
        # divide by execution steps, since if > 1 a batch will have "steps" samples
        self._logger.info('Measured inference times (raw): %s', rstr(inference_time_cb.logs))
        return mean(inference_time_cb.logs[3:]) / self.execution_steps

    def train_model(self, cell_spec: CellSpecification, model_logdir: str):
        '''
        Generate and train the model on all the dataset folds, returning the results of each training session.

        Args:
            cell_spec: cell specification
            model_logdir: log path for saving tensorboard callbacks and other data related to the model

        Returns:
            the model, the training results for each session performed, and the training time for each session
        '''
        # store training results for each fold
        times = np.empty(shape=self.dataset_folds_count, dtype=np.float64)
        histories = []

        for i, (train_ds, val_ds) in enumerate(self.dataset_folds):
            if self.dataset_folds_count > 1:
                self._logger.info("Training on dataset #%d / #%d", i + 1, self.dataset_folds_count)

            # generate a CNN model given the cell specification
            # TODO: instead of rebuilding the model it should be better to just reset the weights and the optimizer
            model, callbacks = self.__compile_model(cell_spec, model_logdir)
            fold_suffix = '' if i == 0 else f'fold_{str(i)}'
            run_name = f'search_{self.current_network_id}{fold_suffix}'
            neptune_run, callbacks = log_service.generate_neptune_run(run_name, ['search'], cell_spec, callbacks)

            # add callback to register as accurate as possible the training time.
            # it must be the first callback, so that it registers the time before other callbacks are executed, registering "almost" only the
            # time due to the training operations (some keras callbacks could be executed before this).
            time_cb = TrainingTimeCallback()
            callbacks.insert(0, time_cb)

            hist = model.fit(x=train_ds,
                             epochs=self.epochs,
                             steps_per_epoch=self.train_batches,
                             validation_data=val_ds,
                             validation_steps=self.validation_batches,
                             callbacks=callbacks,
                             class_weight=self.balanced_class_weights[i])  # type: History

            times[i] = time_cb.get_total_time()
            histories.append(hist.history)

            log_service.finalize_neptune_run(neptune_run, model.count_params(), times[i])

        return model, histories, times
