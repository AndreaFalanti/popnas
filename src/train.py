import csv
import os
import pickle
import statistics
from timeit import default_timer as timer
from typing import Any

import tensorflow as tf

import log_service
import plotter
from controller import ControllerManager
from encoder import SearchSpace
from manager import NetworkManager
from predictors import *
from utils.feature_utils import build_time_feature_names, initialize_features_csv_files, \
    generate_dynamic_reindex_function, build_acc_feature_names, \
    generate_time_features, generate_acc_features
from utils.func_utils import get_valid_inputs_for_block_size, cell_spec_to_str
from utils.nn_utils import TrainingResults
from utils.restore import RestoreInfo, restore_dynamic_reindex_function, restore_train_info, \
    restore_search_space_children


class Train:

    def __init__(self, run_config: 'dict[str, Any]', train_strategy: tf.distribute.Strategy):
        self._logger = log_service.get_logger(__name__)
        self._start_time = timer()

        # search space parameters
        ss_config = run_config['search_space']
        self.blocks = ss_config['blocks']
        self.input_lookback_depth = ss_config['lookback_depth']
        self.operators = ss_config['operators']

        # search strategy parameters
        sstr_config = run_config['search_strategy']
        self.children_max_size = sstr_config['max_children']
        self.pareto_objectives = sstr_config['pareto_objectives']

        # dataset parameters
        ds_config = run_config['dataset']

        # CNN models hyperparameters
        cnn_config = run_config['cnn_hp']

        # CNN architecture parameters
        arc_config = run_config['architecture_parameters']
        max_cells = arc_config['motifs'] * (arc_config['normal_cells_per_motif'] + 1) - 1
        self.multi_output_models = arc_config['multi_output']

        # Other parameters (last category including heterogeneous parameters not classifiable in previous sections)
        others_config = run_config['others']
        self.pnas_mode = others_config['pnas_mode']
        self.preds_batch_size = others_config['predictions_batch_size']
        self.acc_predictor_ensemble_units = others_config['accuracy_predictor_ensemble_units']

        self.rnn_config = run_config.get('rnn_hp')  # None if not defined in config

        # build a search space
        self.search_space = SearchSpace(ss_config, max_cells)

        # create the Network Manager
        self.cnn_manager = NetworkManager(ds_config, cnn_config, arc_config, train_strategy,
                                          others_config['save_children_weights'], others_config['save_children_as_onnx'])

        plotter.initialize_logger()

        restore_info_save_path = log_service.build_path('restore', 'info.pickle')
        if os.path.exists(restore_info_save_path):
            with open(restore_info_save_path, 'rb') as f:
                self.restore_info = pickle.load(f)
        else:
            self.restore_info = RestoreInfo(restore_info_save_path)

        # if not restoring from save, will be initialized to correct values for starting the entire procedure
        restore_data = self.restore_info.get_info()
        self.starting_b = restore_data['current_b']
        self.restore_pareto_train_index = restore_data['pareto_training_index']
        self.restore_exploration_train_index = restore_data['exploration_training_index']
        self.time_delta = restore_data['total_time']

        if self.restore_info.must_restore_dynamic_reindex_function():
            op_times, initial_thrust_time = restore_dynamic_reindex_function()
            reindex_function = generate_dynamic_reindex_function(op_times, initial_thrust_time)
            self.search_space.add_operator_encoder('dynamic_reindex', fn=reindex_function)

        if self.restore_info.must_restore_search_space_children():
            restore_search_space_children(self.search_space, self.starting_b, self.children_max_size, self.pnas_mode)

        # create the predictors
        acc_pred_func, time_pred_func = self.initialize_predictors(train_strategy)

        # set controller step to the correct one (in case of restore is not b=1)
        controller_b = self.starting_b if self.starting_b > 1 else 1
        self.controller = ControllerManager(self.search_space, sstr_config, others_config,
                                            acc_pred_func, time_pred_func,
                                            graph_generator=self.cnn_manager.graph_gen, current_b=controller_b)

    def _compute_total_time(self):
        return self.time_delta + (timer() - self._start_time)

    def generate_and_train_model_from_spec(self, cell_spec: list):
        """
        Generate a model from the cell specification and train it to get an estimate of its quality and characteristics.

        Args:
            cell_spec (list): plain cell specification
        """
        # print the cell in a more comprehensive way
        self.search_space.print_cell_spec(cell_spec)

        # save model if it's the last training batch (full blocks)
        last_block_train = len(cell_spec) == self.blocks
        # build a model, train and get reward and accuracy from the network manager
        train_res = self.cnn_manager.perform_proxy_training(cell_spec, save_best_model=last_block_train)

        self._logger.info("Best accuracy reached: %0.6f", train_res.accuracy)
        self._logger.info("Best F1 score reached: %0.6f", train_res.f1_score)
        self._logger.info("Training time: %0.6f", train_res.training_time)
        self._logger.info('Inference time: %0.6f', train_res.inference_time)
        # format is a workaround for thousands separator, since the python logger has no such feature 
        self._logger.info("Total parameters: %s", format(train_res.params, ','))
        self._logger.info("Total FLOPS: %s", format(train_res.flops, ','))

        return train_res

    def perform_initial_thrust(self, acc_features_len: int):
        '''
        Build a starting point model with 0 blocks to evaluate the offset (initial thrust).
        '''

        self._logger.info('Performing initial thrust with empty cell')
        train_res = self.generate_and_train_model_from_spec([])

        # last fields are exploration and data augmentation
        time_data = [train_res.training_time] + [0, 0, 0, 0, 1, 0, 0, 0, 1] + [False]
        acc_data = [train_res.accuracy] + [0] * (acc_features_len - 3) + [False, False]

        with open(log_service.build_path('csv', 'training_time.csv'), mode='a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(time_data)

        with open(log_service.build_path('csv', 'training_accuracy.csv'), mode='a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(acc_data)

        self.write_training_results_into_csv(train_res)

        return train_res.training_time

    def write_overall_cnn_training_results(self, blocks, times, rewards):
        with open(log_service.build_path('csv', 'training_overview.csv'), mode='a+', newline='') as f:
            writer = csv.writer(f)

            # append mode, so if file handler is in position 0 it means is empty. In this case write the headers too
            if f.tell() == 0:
                writer.writerow(['# blocks', 'avg training time(s)', 'max time', 'min time', 'avg val acc', 'max acc', 'min acc'])

            avg_time = statistics.mean(times)
            max_time = max(times)
            min_time = min(times)

            avg_acc = statistics.mean(rewards)
            max_acc = max(rewards)
            min_acc = min(rewards)

            writer.writerow([blocks, avg_time, max_time, min_time, avg_acc, max_acc, min_acc])

    def write_predictors_training_data(self, current_blocks: int, time: float, accuracy: float, cell_spec: list, exploration: bool = False):
        '''
        Write the training results of a sampled architecture, as formatted features that will be used for training the predictors.

        Args:
            current_blocks (int): [description]
            time (float): [description]
            accuracy (float):
            cell_spec (list): [description]
            exploration:
        '''
        # single record, no data augmentation needed
        time_row = [time] + generate_time_features(cell_spec, self.search_space) + [exploration]

        # accuracy features need data augmentation to generalize on equivalent cell specifications
        eqv_cells, _ = self.search_space.generate_eqv_cells(cell_spec, size=self.blocks)
        # expand cell_spec for bool comparison of data_augmented field
        full_cell_spec = cell_spec + [(None, None, None, None)] * (self.blocks - current_blocks)

        acc_rows = [[accuracy] + generate_acc_features(eqv_cell, self.search_space) + [exploration, eqv_cell != full_cell_spec]
                    for eqv_cell in eqv_cells]

        with open(log_service.build_path('csv', 'training_time.csv'), mode='a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(time_row)

        with open(log_service.build_path('csv', 'training_accuracy.csv'), mode='a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(acc_rows)

    def write_training_results_into_csv(self, train_res: TrainingResults, exploration: bool = False):
        '''
        Append info about a single CNN training to the results csv file.
        '''
        with open(log_service.build_path('csv', 'training_results.csv'), mode='a+', newline='') as f:
            writer = csv.writer(f)

            # append mode, so if file handler is in position 0 it means is empty. In this case write the headers too
            if f.tell() == 0:
                writer.writerow(train_res.get_csv_headers() + ['exploration'])

            # trim cell structure from csv list and replace it with a valid string representation of it
            cell_structure_str = cell_spec_to_str(train_res.cell_spec)
            data = train_res.to_csv_list()[:-1] + [cell_structure_str, exploration]

            writer.writerow(data)

    def initialize_predictors(self, train_strategy: tf.distribute.Strategy):
        acc_col = 'best val accuracy'
        acc_domain = (0, 1)
        predictors_log_path = log_service.build_path('predictors')
        catboost_time_desc_path = log_service.build_path('csv', 'column_desc_time.csv')
        amllibrary_config_path = os.path.join('configs', 'regressors_hyperopt.ini')

        self._logger.info('Initializing predictors...')

        # accuracy predictors to be used
        acc_lstm = AttentionRNNPredictor(self.search_space, acc_col, acc_domain, train_strategy,
                                         self._logger, predictors_log_path, override_logs=False,
                                         save_weights=True, hp_config=self.rnn_config)

        # time predictors to be used
        if not self.pnas_mode:
            time_catboost = CatBoostPredictor(catboost_time_desc_path, self._logger, predictors_log_path, use_random_search=True, override_logs=False)
            # time_lrridge = AMLLibraryPredictor(amllibrary_config_path, ['LRRidge'], self._logger, predictors_log_path, override_logs=False)

        def get_acc_predictor_for_b(b: int):
            return acc_lstm

        def get_time_predictor_for_b(b: int):
            return None if self.pnas_mode else time_catboost

        self._logger.info('Predictors generated successfully')

        return get_acc_predictor_for_b, get_time_predictor_for_b

    def write_smb_times(self, monoblocks_train_info: 'list[tuple[float, float, list[tuple]]]'):
        # dictionary to store specular monoblock (-1 input) times for dynamic reindex
        op_times = {}

        for time, _, cell_spec in monoblocks_train_info:
            # unpack the block (only tuple present in the list) into its components
            in1, op1, in2, op2 = cell_spec[0]

            # get required data for dynamic reindex
            # op_times will contain training times for blocks with both same operation and input -1, for each operation, in order
            same_inputs = in1 == in2
            same_op = op1 == op2
            if same_inputs and same_op and in1 == -1:
                with open(log_service.build_path('csv', 'reindex_op_times.csv'), mode='a+', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([time, op1])
                    op_times[op1] = time

        return op_times

    def log_run_final_results(self):
        self._logger.info('Finished!')
        total_time = self._compute_total_time()

        training_results_path = log_service.build_path('csv', 'training_results.csv')
        with open(training_results_path) as f:
            trained_cnn_count = len(f.readlines())

        self._logger.info('%s', '*' * 40 + ' RUN RESULTS ' + '*' * 40)
        self._logger.info('Trained networks: %d', trained_cnn_count)
        self._logger.info('Total run time: %0.1f seconds (%d hours %d minutes %d seconds)', total_time,
                          total_time // 3600, (total_time // 60) % 60, total_time % 60)
        self._logger.info('*' * 94)

    def generate_final_plots(self):
        ''' Generate plots after the run has been terminated, analyzing the whole results. '''
        plotter.plot_training_info_per_block()
        plotter.plot_cnn_train_boxplots_per_block(self.blocks)
        plotter.plot_predictions_error(self.blocks, self.children_max_size, self.pnas_mode, time_predictor_enabled='time' in self.pareto_objectives)
        plotter.plot_correlations_with_training_time()

        if not self.pnas_mode:
            plotter.plot_pareto_front_curves(self.blocks, self.pareto_objectives)
            plotter.plot_predictions_with_pareto_analysis(self.blocks, self.pareto_objectives)
        if self.multi_output_models:
            plotter.plot_multi_output_boxplot()

    def train_selected_architectures(self, cnns_train_info: 'list[tuple[float, float, list[tuple]]]', current_blocks: int, exploration: bool):
        cell_specs = self.search_space.exploration_front if exploration else self.search_space.children
        restore_index = self.restore_exploration_train_index if exploration else self.restore_pareto_train_index

        for model_index, cell_spec in enumerate(cell_specs):
            # skip networks already trained when restoring a run
            if model_index < restore_index:
                continue

            self._logger.info("%s #%d / #%d", 'Exploration model' if exploration else 'Model', model_index + 1, len(cell_specs))
            self._logger.debug("\t%s", cell_spec)

            train_res = self.generate_and_train_model_from_spec(cell_spec)
            cnns_train_info.append(train_res.to_legacy_info_tuple())
            self._logger.info("Finished %d out of %d %s!", model_index + 1, len(cell_specs), 'exploration models' if exploration else 'models')

            self.write_training_results_into_csv(train_res, exploration=exploration)

            # if current_blocks > 1 we have already the dynamic reindex function, so it's possible to write the feature data immediately
            if current_blocks > 1:
                self.write_predictors_training_data(current_blocks, *train_res.to_legacy_info_tuple(), exploration=exploration)

            if exploration:
                self.restore_info.update(exploration_training_index=model_index + 1, total_time=self._compute_total_time())
            else:
                self.restore_info.update(pareto_training_index=model_index + 1, total_time=self._compute_total_time())

    def process(self):
        '''
        Main function, executed by run.py to start POPNAS algorithm.
        '''
        time_headers, time_feature_types = build_time_feature_names()
        acc_headers, acc_feature_types = build_acc_feature_names(self.blocks, self.input_lookback_depth)

        self.cnn_manager.bootstrap_dataset_lazy_initialization()

        initial_thrust_time = 0
        # if B = 0, perform initial thrust before starting actual training procedure
        if self.starting_b == 0:
            # add headers to csv and create CatBoost feature files
            initialize_features_csv_files(time_headers, time_feature_types, acc_headers, acc_feature_types, log_service.build_path('csv'))

            initial_thrust_time = self.perform_initial_thrust(len(acc_headers))
            self.starting_b = 1
            self.restore_info.update(current_b=1, total_time=self._compute_total_time())

        # train the selected CNN networks for each number of blocks (algorithm step)
        for current_blocks in range(self.starting_b, self.blocks + 1):
            cnns_train_info = [] if self.restore_pareto_train_index == 0 \
                else restore_train_info(current_blocks)  # type: list[tuple[float, float, list[tuple]]]

            self.train_selected_architectures(cnns_train_info, current_blocks, exploration=False)
            # train the models built from exploration pareto front
            self.train_selected_architectures(cnns_train_info, current_blocks, exploration=True)

            # all CNN with current_blocks = 1 have been trained, build the dynamic reindex and write the feature dataset for regressors
            if current_blocks == 1:
                op_times = self.write_smb_times(cnns_train_info)
                reindex_function = generate_dynamic_reindex_function(op_times, initial_thrust_time)
                self.search_space.add_operator_encoder('dynamic_reindex', fn=reindex_function)
                plotter.plot_smb_info()

                for time, acc, cell_spec in cnns_train_info:
                    self.write_predictors_training_data(current_blocks, time, acc, cell_spec)

            times, rewards, _ = zip(*cnns_train_info)
            self.write_overall_cnn_training_results(current_blocks, times, rewards)

            # perform controller training, pareto front estimation and plots building if not at final step
            if current_blocks != self.blocks:
                self.controller.train_step()
                self.controller.update_step()

                # controller new cells have 1 more block
                expansion_step_blocks = current_blocks + 1
                valid_inputs = get_valid_inputs_for_block_size(self.search_space.input_values, current_blocks=expansion_step_blocks,
                                                               max_blocks=self.blocks)

                # PNAS mode doesn't build pareto front
                if not self.pnas_mode:
                    plotter.plot_pareto_inputs_and_operators_usage(expansion_step_blocks, self.operators, valid_inputs, limit=self.children_max_size)
                    plotter.plot_exploration_inputs_and_operators_usage(expansion_step_blocks, self.operators, valid_inputs)

                # state_space.children are updated in controller.update_step, CNN to train in next step. Add also exploration networks.
                trained_cells = self.search_space.children + self.search_space.exploration_front
                plotter.plot_children_inputs_and_operators_usage(expansion_step_blocks, self.operators, valid_inputs, trained_cells)

                self.restore_info.update(current_b=expansion_step_blocks, pareto_training_index=0, exploration_training_index=0,
                                         total_time=self._compute_total_time())
                self.restore_pareto_train_index = 0
                self.restore_exploration_train_index = 0

        self.generate_final_plots()
        self.log_run_final_results()
