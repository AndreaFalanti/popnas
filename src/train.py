import csv
import importlib.util
import os
import statistics

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.datasets import cifar100

import log_service
import plotter
from controller import ControllerManager
from encoder import StateSpace
from manager import NetworkManager
from predictors import *
from utils.feature_utils import *
from utils.func_utils import get_valid_inputs_for_block_size

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # disable Tensorflow info messages


class Train:

    def __init__(self, blocks, children_max_size, exploration_max_size,
                 dataset, sets,
                 epochs, batch_size, learning_rate, filters, weight_reg,
                 cell_stacks, normal_cells_per_stack,
                 all_blocks_concat, pnas_mode):

        self._logger = log_service.get_logger(__name__)

        # search space parameters
        self.blocks = blocks
        self.children_max_size = children_max_size
        self.exploration_max_size = exploration_max_size
        self.input_lookback_depth = -2

        # dataset parameters
        self.dataset = dataset
        self.sets = sets

        # CNN models parameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.filters = filters
        self.weight_reg = weight_reg
        self.concat_only_unused = not all_blocks_concat
        self.cell_stacks = cell_stacks
        self.normal_cells_per_stack = normal_cells_per_stack

        self.max_cells = cell_stacks * (normal_cells_per_stack + 1) - 1

        self.pnas_mode = pnas_mode

        plotter.initialize_logger()

    def load_dataset(self):
        if self.dataset == "cifar10":
            (x_train_init, y_train_init), (x_test_init, y_test_init) = cifar10.load_data()
        elif self.dataset == "cifar100":
            (x_train_init, y_train_init), (x_test_init, y_test_init) = cifar100.load_data()
        # TODO: untested legacy code, not sure this is still working
        else:
            spec = importlib.util.spec_from_file_location("dataset", self.dataset)
            dataset = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(dataset)
            (x_train_init, y_train_init), (x_test_init, y_test_init) = dataset.load_data()

        return (x_train_init, y_train_init), (x_test_init, y_test_init)

    def prepare_dataset(self, x_train_init, y_train_init):
        """Build a validation set from training set and do some preprocessing

        Args:
            x_train_init (ndarray): x training
            y_train_init (ndarray): y training

        Returns:
            list:
        """
        # normalize image RGB values into [0, 1] domain
        x_train_init = x_train_init.astype('float32') / 255.

        datasets = []
        # TODO: why using a dataset multiple times if sets > 1? Is this actually useful or it's possible to deprecate this feature?
        # TODO: splits for other datasets are actually not defined
        for i in range(0, self.sets):
            # TODO: take only 10000 images for fast training (one batch of cifar10), make it random in future?
            # limit = 10000
            # x_train_init = x_train_init[:limit]
            # y_train_init = y_train_init[:limit]

            # create a validation set for evaluation of the child models
            x_train, x_validation, y_train, y_validation = train_test_split(x_train_init, y_train_init, test_size=0.1, random_state=0,
                                                                            stratify=y_train_init)

            if self.dataset == "cifar10":
                # cifar10
                y_train = to_categorical(y_train, 10)
                y_validation = to_categorical(y_validation, 10)

            elif self.dataset == "cifar100":
                # cifar100
                y_train = to_categorical(y_train, 100)
                y_validation = to_categorical(y_validation, 100)

            # TODO: logic is missing for custom dataset usage

            # pack the dataset for the NetworkManager
            datasets.append([x_train, y_train, x_validation, y_validation])

        return datasets

    def generate_and_train_model_from_spec(self, state_space: StateSpace, manager: NetworkManager, cell_spec: list):
        """
        Generate a model given the actions and train it to get reward and time

        Args:
            state_space (StateSpace): ...
            manager (NetworkManager): ...
            cell_spec (list): plain cell specification

        Returns:
            tuple: 4 elements: (max accuracy, training time, params, flops) of trained CNN
        """
        # print the cell in a more comprehensive way
        state_space.print_cell_spec(cell_spec)

        # save model if it's the last training batch (full blocks)
        last_block_train = len(cell_spec) == self.blocks
        # build a model, train and get reward and accuracy from the network manager
        reward, timer, total_params, flops = manager.get_rewards(cell_spec, save_best_model=last_block_train)

        self._logger.info("Best accuracy reached: %0.6f", reward)
        self._logger.info("Training time: %0.6f", timer)
        # format is a workaround for thousands separator, since the python logger has no such feature 
        self._logger.info("Total parameters: %s", format(total_params, ','))
        self._logger.info("Total FLOPS: %s", format(flops, ','))

        return reward, timer, total_params, flops

    def perform_initial_thrust(self, state_space: StateSpace, manager: NetworkManager, time_features_len: int, acc_features_len: int):
        '''
        Build a starting point model with 0 blocks to evaluate the offset (initial thrust).

        Args:
            state_space (StateSpace): [description]
            manager (NetworkManager): [description]
        '''

        self._logger.info('Performing initial thrust with empty cell')
        acc, time, params, flops = self.generate_and_train_model_from_spec(state_space, manager, [])

        # last field is data augmentation, True for generated sample only
        time_data = [time] + [0] * (time_features_len - 2) + [False]
        acc_data = [acc] + [0] * (acc_features_len - 2) + [False]

        with open(log_service.build_path('csv', 'training_time.csv'), mode='a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(time_data)

        with open(log_service.build_path('csv', 'training_accuracy.csv'), mode='a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(acc_data)

        self.write_training_results_into_csv([], acc, time, params, flops, 0)

        return time

    def write_overall_cnn_training_results(self, blocks, timers, rewards):
        with open(log_service.build_path('csv', 'training_overview.csv'), mode='a+', newline='') as f:
            writer = csv.writer(f)

            # append mode, so if file handler is in position 0 it means is empty. In this case write the headers too
            if f.tell() == 0:
                writer.writerow(['# blocks', 'avg training time(s)', 'max time', 'min time', 'avg val acc', 'max acc', 'min acc'])

            avg_time = statistics.mean(timers)
            max_time = max(timers)
            min_time = min(timers)

            avg_acc = statistics.mean(rewards)
            max_acc = max(rewards)
            min_acc = min(rewards)

            writer.writerow([blocks, avg_time, max_time, min_time, avg_acc, max_acc, min_acc])

    def generate_eqv_cells_features(self, current_blocks: int, time: float, accuracy: float, cell_spec: list,
                                    state_space: StateSpace, exploration: bool):
        '''
        Builds all the allowed permutations of the blocks present in the cell, which are the equivalent encodings.
        Then, for each equivalent cell, produce the features set for both time and accuracy predictors.

        Returns:
            (list): features to be used in predictors (ML techniques)
        '''

        # equivalent cells can be useful to train better the regressor
        eqv_cells, _ = state_space.generate_eqv_cells(cell_spec, size=self.blocks)

        # expand cell_spec for bool comparison of data_augmented field
        cell_spec = cell_spec + [(None, None, None, None)] * (self.blocks - current_blocks)

        time_features_list, acc_features_list = [], []
        for eqv_cell in eqv_cells:
            time_features, acc_features = generate_all_feature_sets(cell_spec, state_space)

            # features are expanded with labels and data_augmented field
            time_features_list.append([time] + time_features + [exploration, eqv_cell != cell_spec])
            acc_features_list.append([accuracy] + acc_features + [exploration, eqv_cell != cell_spec])

        return time_features_list, acc_features_list

        # TODO: legacy features
        # return [[timer, current_blocks] + state_space.encode_cell_spec(cell, op_enc_name='dynamic_reindex') + [cell != cell_spec]
        #         for cell in eqv_cells],\
        #        [[accuracy, current_blocks] + state_space.encode_cell_spec(cell) + [cell != cell_spec] for cell in eqv_cells]

    def write_training_data(self, current_blocks: int, timer: float, accuracy: float, cell_spec: list,
                            state_space: StateSpace, exploration: bool = False):
        '''
        Write on csv the training time, that will be used for regressor training, and the accuracy reached, that can be used for controller training.
        Use sliding blocks mechanism and cell equivalence data augmentation to multiply the entries.

        Args:
            current_blocks (int): [description]
            timer (float): [description]
            accuracy (float):
            cell_spec (list): [description]
            state_space (StateSpace): [description]
            exploration:
        '''

        time_rows, acc_rows = self.generate_eqv_cells_features(current_blocks, timer, accuracy, cell_spec, state_space, exploration)

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

    def initialize_predictors(self, state_space):
        acc_col = 'best val accuracy'
        acc_domain = (0, 1)
        predictors_log_path = log_service.build_path('predictors')
        catboost_time_desc_path = log_service.build_path('csv', 'column_desc_time.csv')
        amllibrary_config_path = os.path.join('configs', 'regressors_hyperopt.ini')

        self._logger.info('Initializing predictors...')

        # accuracy predictors to be used
        acc_lstm = LSTMPredictor(state_space, acc_col, acc_domain, self._logger, predictors_log_path,
                                 lr=0.002, weight_reg=1e-6, embedding_dim=20, rnn_cells=100, epochs=20)

        # time predictors to be used
        time_catboost = CatBoostPredictor(catboost_time_desc_path, self._logger, predictors_log_path, use_random_search=True)
        time_xgboost = AMLLibraryPredictor(amllibrary_config_path, ['XGBoost'], self._logger, predictors_log_path)
        time_lrridge = AMLLibraryPredictor(amllibrary_config_path, ['LRRidge'], self._logger, predictors_log_path)

        def get_acc_predictor_for_b(b: int):
            return acc_lstm

        def get_time_predictor_for_b(b: int):
            return time_lrridge if b == 1 else time_catboost

        self._logger.info('Predictors generated successfully')

        return get_acc_predictor_for_b, get_time_predictor_for_b

    def process(self):
        '''
        Main function, executed by run.py to start POPNAS algorithm.
        '''
        # dictionary to store specular monoblock (-1 input) times for dynamic reindex
        op_timers = {}

        # TODO: restore search space
        operators = ['identity', '3x3 dconv', '5x5 dconv', '7x7 dconv', '1x7-7x1 conv', '3x3 conv', '3x3 maxpool', '3x3 avgpool']
        # operators = ['identity', '3x3 dconv']

        starting_b = 0
        self._logger.info('Total cells stacked in each CNN: %d', self.max_cells)

        # construct a state space
        state_space = StateSpace(self.blocks, operators, self.max_cells, input_lookback_depth=self.input_lookback_depth, input_lookforward_depth=None)

        max_lookback_depth = abs(self.input_lookback_depth)
        time_headers, time_feature_types = build_feature_names('time', self.blocks, max_lookback_depth)
        acc_headers, acc_feature_types = build_feature_names('acc', self.blocks, max_lookback_depth)

        # add headers to csv and create CatBoost feature files
        initialize_features_csv_files(time_headers, time_feature_types, acc_headers, acc_feature_types, log_service.build_path('csv'))

        # load correct dataset (based on self.dataset), test data is not used actually
        (x_train_init, y_train_init), _ = self.load_dataset()

        dataset = self.prepare_dataset(x_train_init, y_train_init)

        # create the Network Manager
        manager = NetworkManager(dataset, data_num=self.sets, epochs=self.epochs, batchsize=self.batch_size,
                                 learning_rate=self.learning_rate, filters=self.filters, weight_reg=self.weight_reg,
                                 cell_stacks=self.cell_stacks, normal_cells_per_stack=self.normal_cells_per_stack,
                                 concat_only_unused=self.concat_only_unused)

        # create the predictors
        acc_pred_func, time_pred_func = self.initialize_predictors(state_space)

        # create the ControllerManager and build the internal policy network
        controller = ControllerManager(state_space, acc_pred_func, time_pred_func,
                                       B=self.blocks, K=self.children_max_size, ex=self.exploration_max_size, pnas_mode=self.pnas_mode)

        initial_thrust_time = 0
        # if B = 0, perform initial thrust before starting actual training procedure
        if starting_b == 0:
            initial_thrust_time = self.perform_initial_thrust(state_space, manager, len(time_headers), len(acc_headers))
            starting_b = 1

        monoblocks_info = []

        # train the child CNN networks for each number of blocks
        for current_blocks in range(starting_b, self.blocks + 1):
            rewards = []
            timers = []

            cell_specs = state_space.get_cells_to_train()

            for model_index, cell_spec in enumerate(cell_specs):
                self._logger.info("Model #%d / #%d", model_index + 1, len(cell_specs))
                self._logger.debug("\t%s", cell_spec)

                reward, timer, total_params, flops = self.generate_and_train_model_from_spec(state_space, manager, cell_spec)
                rewards.append(reward)
                timers.append(timer)

                if current_blocks == 1:
                    monoblocks_info.append([timer, reward, cell_spec])
                    # unpack the block (only tuple present in the list) into its components
                    in1, op1, in2, op2 = cell_spec[0]

                    # get required data for dynamic reindex
                    # op_timers will contain timers for blocks with both same operation and input -1, for each operation, in order
                    same_inputs = in1 == in2
                    same_op = op1 == op2
                    if same_inputs and same_op and in1 == -1:
                        with open(log_service.build_path('csv', 'reindex_op_times.csv'), mode='a+', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([timer, op1])
                            op_timers[op1] = timer

                self._logger.info("Finished %d out of %d models!", (model_index + 1), len(cell_specs))

                self.write_training_results_into_csv(cell_spec, reward, timer, total_params, flops, current_blocks)

                # if current_blocks > 1, we have already the dynamic reindex function and it's possible to write the feature data immediately
                if current_blocks > 1:
                    self.write_training_data(current_blocks, timer, reward, cell_spec, state_space)

            # train the models built from exploration pareto front
            for model_index, cell_spec in enumerate(state_space.exploration_front):
                if model_index == 0:
                    self._logger.info('Starting exploration step...')

                self._logger.info("Exploration Model #%d / #%d", model_index + 1, len(state_space.exploration_front))
                self._logger.debug("\t%s", cell_spec)

                reward, timer, total_params, flops = self.generate_and_train_model_from_spec(state_space, manager, cell_spec)
                rewards.append(reward)
                self._logger.info("Finished %d out of %d exploration models!", (model_index + 1), len(state_space.exploration_front))

                self.write_training_results_into_csv(cell_spec, reward, timer, total_params, flops, current_blocks, exploration=True)
                self.write_training_data(current_blocks, timer, reward, cell_spec, state_space, exploration=True)

            # all CNN with current_blocks = 1 have been trained, build the dynamic reindex and write the feature dataset for regressors
            if current_blocks == 1:
                reindex_function = generate_dynamic_reindex_function(op_timers, initial_thrust_time)
                state_space.add_operator_encoder('dynamic_reindex', fn=reindex_function)
                plotter.plot_dynamic_reindex_related_blocks_info()

                for time, acc, cell_spec in monoblocks_info:
                    self.write_training_data(current_blocks, time, acc, cell_spec, state_space)

            self.write_overall_cnn_training_results(current_blocks, timers, rewards)

            # perform controller training, pareto front estimation and plots building if not at final step
            if current_blocks != self.blocks:
                controller.train_step(rewards)
                controller.update_step()

                # controller new cells have 1 more block
                expansion_step_blocks = current_blocks + 1
                valid_inputs = get_valid_inputs_for_block_size(state_space.input_values, current_blocks=expansion_step_blocks, max_blocks=self.blocks)

                # PNAS mode doesn't build pareto front
                if not self.pnas_mode:
                    plotter.plot_pareto_inputs_and_operators_usage(expansion_step_blocks, operators, valid_inputs)
                    plotter.plot_exploration_inputs_and_operators_usage(expansion_step_blocks, operators, valid_inputs)

                # state_space.children are updated in controller.update_step, CNN to train in next step
                plotter.plot_children_inputs_and_operators_usage(expansion_step_blocks, operators, valid_inputs, state_space.children)

        plotter.plot_training_info_per_block()
        plotter.plot_cnn_train_boxplots_per_block(self.blocks)
        plotter.plot_predictions_error(self.blocks, self.pnas_mode)
        if not self.pnas_mode:
            plotter.plot_pareto_front_curves(self.blocks, plot3d=True)

        self._logger.info("Finished!")
