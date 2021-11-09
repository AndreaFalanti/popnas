import csv
import os
import statistics
from timeit import default_timer as timer
from typing import Any

import log_service
import plotter
from controller import ControllerManager
from encoder import SearchSpace
from manager import NetworkManager
from model import ModelGenerator
from predictors import *
from utils.feature_utils import generate_all_feature_sets, build_feature_names, initialize_features_csv_files, generate_dynamic_reindex_function
from utils.func_utils import get_valid_inputs_for_block_size

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # disable Tensorflow info messages


class Train:

    def __init__(self, run_config: 'dict[str, Any]'):

        self._logger = log_service.get_logger(__name__)
        self._start_time = timer()

        # TODO: instantiate relevant classes here and avoid to store unnecessary config parameters
        # search space parameters
        ss_config = run_config['search_space']
        self.blocks = ss_config['blocks']
        self.children_max_size = ss_config['max_children']
        self.exploration_max_size = ss_config['max_exploration_children']
        self.input_lookback_depth = ss_config['lookback_depth']
        self.input_lookforward_depth = ss_config['lookforward_depth']
        self.operators = ss_config['operators']

        # dataset parameters
        ds_config = run_config['dataset']

        # CNN models hyperparameters
        cnn_config = run_config['cnn_hp']

        # CNN architecture parameters
        arc_config = run_config['architecture_parameters']
        max_cells = arc_config['motifs'] * (arc_config['normal_cells_per_motif'] + 1) - 1

        # build a search space
        self.search_space = SearchSpace(ss_config['blocks'], ss_config['operators'], max_cells,
                                        input_lookback_depth=-ss_config['lookback_depth'], input_lookforward_depth=ss_config['lookforward_depth'])

        self.model_gen = ModelGenerator(cnn_config['learning_rate'], cnn_config['filters'], cnn_config['weight_reg'],
                                        arc_config['normal_cells_per_motif'], arc_config['motifs'], arc_config['concat_only_unused_blocks'])

        # create the Network Manager
        self.cnn_manager = NetworkManager(self.model_gen, ds_config, epochs=cnn_config['epochs'], batch_size=cnn_config['batch_size'])

        self.pnas_mode = run_config['pnas_mode']

        self.lstm_config = run_config['lstm_hp']

        plotter.initialize_logger()

    def generate_and_train_model_from_spec(self, cell_spec: list):
        """
        Generate a model given the actions and train it to get reward and time

        Args:
            cell_spec (list): plain cell specification

        Returns:
            tuple: 4 elements: (max accuracy, training time, params, flops) of trained CNN
        """
        # print the cell in a more comprehensive way
        self.search_space.print_cell_spec(cell_spec)

        # save model if it's the last training batch (full blocks)
        last_block_train = len(cell_spec) == self.blocks
        # build a model, train and get reward and accuracy from the network manager
        reward, time, total_params, flops = self.cnn_manager.get_rewards(cell_spec, save_best_model=last_block_train)

        self._logger.info("Best accuracy reached: %0.6f", reward)
        self._logger.info("Training time: %0.6f", time)
        # format is a workaround for thousands separator, since the python logger has no such feature 
        self._logger.info("Total parameters: %s", format(total_params, ','))
        self._logger.info("Total FLOPS: %s", format(flops, ','))

        return reward, time, total_params, flops

    def perform_initial_thrust(self, time_features_len: int, acc_features_len: int):
        '''
        Build a starting point model with 0 blocks to evaluate the offset (initial thrust).
        '''

        self._logger.info('Performing initial thrust with empty cell')
        acc, time, params, flops = self.generate_and_train_model_from_spec([])

        # last fields are exploration and data augmentation
        time_data = [time] + [0] * (time_features_len - 3) + [False, False]
        acc_data = [acc] + [0] * (acc_features_len - 3) + [False, False]

        with open(log_service.build_path('csv', 'training_time.csv'), mode='a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(time_data)

        with open(log_service.build_path('csv', 'training_accuracy.csv'), mode='a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(acc_data)

        self.write_training_results_into_csv([], acc, time, params, flops, 0)

        return time

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

    def generate_eqv_cells_features(self, current_blocks: int, time: float, accuracy: float, cell_spec: list, exploration: bool):
        '''
        Builds all the allowed permutations of the blocks present in the cell, which are the equivalent encodings.
        Then, for each equivalent cell, produce the features set for both time and accuracy predictors.

        Returns:
            (list): features to be used in predictors (ML techniques)
        '''

        # equivalent cells can be useful to train better the regressor
        eqv_cells, _ = self.search_space.generate_eqv_cells(cell_spec, size=self.blocks)

        # expand cell_spec for bool comparison of data_augmented field
        cell_spec = cell_spec + [(None, None, None, None)] * (self.blocks - current_blocks)

        time_features_list, acc_features_list = [], []
        for eqv_cell in eqv_cells:
            time_features, acc_features = generate_all_feature_sets(eqv_cell, self.search_space)

            # features are expanded with labels and data_augmented field
            time_features_list.append([time] + time_features + [exploration, eqv_cell != cell_spec])
            acc_features_list.append([accuracy] + acc_features + [exploration, eqv_cell != cell_spec])

        return time_features_list, acc_features_list

        # TODO: legacy features
        # return [[time, current_blocks] + state_space.encode_cell_spec(cell, op_enc_name='dynamic_reindex') + [cell != cell_spec]
        #         for cell in eqv_cells],\
        #        [[accuracy, current_blocks] + state_space.encode_cell_spec(cell) + [cell != cell_spec] for cell in eqv_cells]

    def write_training_data(self, current_blocks: int, time: float, accuracy: float, cell_spec: list, exploration: bool = False):
        '''
        Write on csv the training time, that will be used for regressor training, and the accuracy reached, that can be used for controller training.
        Use sliding blocks mechanism and cell equivalence data augmentation to multiply the entries.

        Args:
            current_blocks (int): [description]
            time (float): [description]
            accuracy (float):
            cell_spec (list): [description]
            exploration:
        '''

        time_rows, acc_rows = self.generate_eqv_cells_features(current_blocks, time, accuracy, cell_spec, exploration)

        with open(log_service.build_path('csv', 'training_time.csv'), mode='a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(time_rows)

        with open(log_service.build_path('csv', 'training_accuracy.csv'), mode='a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(acc_rows)

    def write_training_results_into_csv(self, cell_spec: list, acc: float, time: float, params: int,
                                        flops: int, blocks: int, exploration: bool = False):
        '''
        Append info about a single CNN training to the results csv file.
        '''
        with open(log_service.build_path('csv', 'training_results.csv'), mode='a+', newline='') as f:
            writer = csv.writer(f)

            # append mode, so if file handler is in position 0 it means is empty. In this case write the headers too
            if f.tell() == 0:
                writer.writerow(['best val accuracy', 'training time(seconds)', 'total params', 'flops', '# blocks', 'exploration', 'cell structure'])

            cell_structure = f"[{';'.join(map(lambda el: str(el), cell_spec))}]"
            data = [acc, time, params, flops, blocks, exploration, cell_structure]

            writer.writerow(data)

    def initialize_predictors(self):
        acc_col = 'best val accuracy'
        acc_domain = (0, 1)
        predictors_log_path = log_service.build_path('predictors')
        catboost_time_desc_path = log_service.build_path('csv', 'column_desc_time.csv')
        amllibrary_config_path = os.path.join('configs', 'regressors_hyperopt.ini')

        self._logger.info('Initializing predictors...')

        # accuracy predictors to be used
        acc_lstm = LSTMPredictor(self.search_space, acc_col, acc_domain, self._logger, predictors_log_path,
                                 lr=self.lstm_config['learning_rate'], weight_reg=self.lstm_config['weight_reg'],
                                 embedding_dim=self.lstm_config['embedding_dim'], rnn_cells=self.lstm_config['cells'],
                                 epochs=self.lstm_config['epochs'])

        # time predictors to be used
        if not self.pnas_mode:
            # TODO: shap (0.40.0) is bugged, avoid feature analysis for now since the local fixes can't be easily replicated on all servers on which
            #  the algorithm is ran. Hopefully pull requests will be merged in near future.
            time_catboost = CatBoostPredictor(catboost_time_desc_path, self._logger, predictors_log_path, use_random_search=True,
                                              perform_feature_analysis=False)
            time_xgboost = AMLLibraryPredictor(amllibrary_config_path, ['XGBoost'], self._logger, predictors_log_path, perform_feature_analysis=False)
            time_lrridge = AMLLibraryPredictor(amllibrary_config_path, ['LRRidge'], self._logger, predictors_log_path, perform_feature_analysis=False)

        def get_acc_predictor_for_b(b: int):
            return acc_lstm

        def get_time_predictor_for_b(b: int):
            return None if self.pnas_mode else (time_lrridge if b == 1 else time_catboost)

        self._logger.info('Predictors generated successfully')

        return get_acc_predictor_for_b, get_time_predictor_for_b

    def write_smb_results(self, monoblocks_train_info: 'list[tuple[float, float, list[tuple]]]'):
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
        total_time = timer() - self._start_time

        training_results_path = log_service.build_path('csv', 'training_results.csv')
        with open(training_results_path) as f:
            trained_cnn_count = len(f.readlines())

        self._logger.info('%s', '*' * 40 + ' RUN RESULTS ' + '*' * 40)
        self._logger.info('Trained networks: %d', trained_cnn_count)
        self._logger.info('Total run time: %0.1f seconds (%d hours %d minutes %d seconds)', total_time,
                          total_time // 3600, (total_time // 60) % 60, total_time % 60)
        self._logger.info('*' * 94)

    def process(self):
        '''
        Main function, executed by run.py to start POPNAS algorithm.
        '''
        starting_b = 0

        time_headers, time_feature_types = build_feature_names('time', self.blocks, self.input_lookback_depth)
        acc_headers, acc_feature_types = build_feature_names('acc', self.blocks, self.input_lookback_depth)

        # add headers to csv and create CatBoost feature files
        initialize_features_csv_files(time_headers, time_feature_types, acc_headers, acc_feature_types, log_service.build_path('csv'))

        # create the predictors
        acc_pred_func, time_pred_func = self.initialize_predictors()

        # create the ControllerManager and build the internal policy network
        controller = ControllerManager(self.search_space, acc_pred_func, time_pred_func,
                                       B=self.blocks, K=self.children_max_size, ex=self.exploration_max_size, pnas_mode=self.pnas_mode)

        initial_thrust_time = 0
        # if B = 0, perform initial thrust before starting actual training procedure
        if starting_b == 0:
            initial_thrust_time = self.perform_initial_thrust(len(time_headers), len(acc_headers))
            starting_b = 1

        # train the child CNN networks for each number of blocks
        for current_blocks in range(starting_b, self.blocks + 1):
            cnns_train_info = []  # type: list[tuple[float, float, list[tuple]]]

            for model_index, cell_spec in enumerate(self.search_space.children):
                self._logger.info("Model #%d / #%d", model_index + 1, len(self.search_space.children))
                self._logger.debug("\t%s", cell_spec)

                reward, time, total_params, flops = self.generate_and_train_model_from_spec(cell_spec)
                cnns_train_info.append((time, reward, cell_spec))
                self._logger.info("Finished %d out of %d models!", model_index + 1, len(self.search_space.children))

                self.write_training_results_into_csv(cell_spec, reward, time, total_params, flops, current_blocks)

                # if current_blocks > 1, we have already the dynamic reindex function and it's possible to write the feature data immediately
                if current_blocks > 1:
                    self.write_training_data(current_blocks, time, reward, cell_spec)

            # train the models built from exploration pareto front
            for model_index, cell_spec in enumerate(self.search_space.exploration_front):
                if model_index == 0:
                    self._logger.info('Starting exploration step...')

                self._logger.info("Exploration Model #%d / #%d", model_index + 1, len(self.search_space.exploration_front))
                self._logger.debug("\t%s", cell_spec)

                reward, time, total_params, flops = self.generate_and_train_model_from_spec(cell_spec)
                cnns_train_info.append((time, reward, cell_spec))
                self._logger.info("Finished %d out of %d exploration models!", model_index + 1, len(self.search_space.exploration_front))

                self.write_training_results_into_csv(cell_spec, reward, time, total_params, flops, current_blocks, exploration=True)
                self.write_training_data(current_blocks, time, reward, cell_spec, exploration=True)

            # all CNN with current_blocks = 1 have been trained, build the dynamic reindex and write the feature dataset for regressors
            if current_blocks == 1:
                op_times = self.write_smb_results(cnns_train_info)
                reindex_function = generate_dynamic_reindex_function(op_times, initial_thrust_time)
                self.search_space.add_operator_encoder('dynamic_reindex', fn=reindex_function)
                plotter.plot_dynamic_reindex_related_blocks_info()

                for time, acc, cell_spec in cnns_train_info:
                    self.write_training_data(current_blocks, time, acc, cell_spec)

            times, rewards, _ = zip(*cnns_train_info)
            self.write_overall_cnn_training_results(current_blocks, times, rewards)

            # perform controller training, pareto front estimation and plots building if not at final step
            if current_blocks != self.blocks:
                controller.train_step(rewards)
                controller.update_step()

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

        plotter.plot_training_info_per_block()
        plotter.plot_cnn_train_boxplots_per_block(self.blocks)
        plotter.plot_predictions_error(self.blocks, self.pnas_mode)
        if not self.pnas_mode:
            plotter.plot_pareto_front_curves(self.blocks, plot3d=True)

        self._logger.info("Finished!")
        self.log_run_final_results()
