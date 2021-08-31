import log_service
from model import ModelGenerator
from manager import NetworkManager
from controller import ControllerManager
from encoder import StateSpace

from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.datasets import cifar100
from tensorflow.python.keras.datasets import cifar10
import numpy as np
from sklearn.model_selection import train_test_split

import importlib.util
import csv
import statistics

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # disable GPU debugging info

class Train:

    def __init__(self, blocks, children, checkpoint,
                 dataset, sets, epochs, batchsize,
                 learning_rate, restore, filters,
                 all_blocks_concat, pnas_mode):

        self._logger = log_service.get_logger(__name__)

        self.blocks = blocks
        self.checkpoint = checkpoint
        self.children = children
        self.dataset = dataset
        self.sets = sets
        self.epochs = epochs
        self.batchsize = batchsize
        self.learning_rate = learning_rate
        self.restore = restore
        self.filters = filters
        self.concat_only_unused = not all_blocks_concat
        self.pnas_mode = pnas_mode

    def load_dataset(self):
        if self.dataset == "cifar10":
            (x_train_init, y_train_init), (x_test_init, y_test_init) = cifar10.load_data()
        elif self.dataset == "cifar100":
            (x_train_init, y_train_init), (x_test_init, y_test_init) = cifar100.load_data()
        else:
            spec = importlib.util.spec_from_file_location("dataset", self.dataset)
            set = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(set)
            (x_train_init, y_train_init), (x_test_init, y_test_init) = set.load_data()

        return (x_train_init, y_train_init), (x_test_init, y_test_init)

    def prepare_dataset(self, x_train_init, y_train_init, x_test, y_test):
        """Build a validation set from training set and do some preprocessing

        Args:
            x_train_init (ndarray): x training
            y_train_init (ndarray): y training
            x_test (ndarray): x test
            y_test (ndarray): y test

        Returns:
            list:
        """
        # normalize image RGB values into [0, 1] domain
        x_train_init = x_train_init.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.

        dataset = []
        # TODO: why using a dataset multiple times if sets > 1? Is this actually useful or its possible to deprecate this feature?
        # TODO: splits for other datasets are actually not defined
        for i in range(0, self.sets):
            # TODO: take only 10000 images for fast training (one batch of cifar10), make it random in future?
            limit = 10000
            x_train_init = x_train_init[:limit]
            y_train_init = y_train_init[:limit]

            # create a validation set for evaluation of the child models
            x_train, x_validation, y_train, y_validation = train_test_split(x_train_init, y_train_init, test_size=0.1, random_state=0)

            # TODO: missing y_test to categorical or done somewhere else?
            if self.dataset == "cifar10":
                # cifar10
                y_train = to_categorical(y_train, 10)
                y_validation = to_categorical(y_validation, 10)

            elif self.dataset == "cifar100":
                # cifar100
                y_train = to_categorical(y_train, 100)
                y_validation = to_categorical(y_validation, 100)

            # TODO: if custom dataset is used, all works fine or some logic is missing?

            # pack the dataset for the NetworkManager
            dataset.append([x_train, y_train, x_validation, y_validation])

        return dataset

    def generate_and_train_model_from_actions(self, state_space, manager, actions, model_index, total_models_for_step):
        """
        Generate a model given the actions and train it to get reward and time

        Args:
            state_space (StateSpace): ...
            manager (NetworkManager): ...
            actions (type): actions embedding (oneHot)
            model_index (int): model number (for logging only)
            total_models_for_step (int): total amount of models to train in this step (for logging only)

        Returns:
            tuple: reward, timer, listed_space(parsed actions embedding)
        """
        # print the action probabilities
        state_space.print_actions(actions)
        listed_space = state_space.parse_state_space_list(actions)
        self._logger.info("Model #%d / #%d", model_index, total_models_for_step)
        self._logger.info("\t%s", listed_space)

        # build a model, train and get reward and accuracy from the network manager
        reward, timer = manager.get_rewards(ModelGenerator, listed_space, self.concat_only_unused, save_best_model=(len(actions) // 4 == self.blocks))
        self._logger.info("Final Accuracy: %0.6f", reward)
        self._logger.info("Training time: %0.6f", timer)

        return reward, timer, listed_space

    def generate_dynamic_reindex_function(self, operators, op_timers, t_max):
        '''
        Closure for generating a function to easily apply dynamic reindex where necessary.

        Args:
            operators (list<str>): allowed operations
            op_timers (list<float>): timers for each block with same operations, in order
            t_max (float): max time between op_timers

        Returns:
            Callable(int): dynamic reindex function
        '''

        def apply_dynamic_reindex(op_index):
            return len(operators) * op_timers[op_index] / t_max

        return apply_dynamic_reindex

    def perform_initial_thrust(self, state_space, manager):
        '''
        Build a starting point model with 0 blocks to evaluate the offset (initial thrust).

        Args:
            state_space ([type]): [description]
            manager ([type]): [description]
        '''

        reward, timer, _ = self.generate_and_train_model_from_actions(state_space, manager, [], 1, 1)

        with open(log_service.build_path('csv', 'training_time.csv'), mode='a+', newline='') as f:
            data = [timer, 0]
            for i in range(self.blocks):
                data.extend([0, 0, 0, 0])
            writer = csv.writer(f)
            writer.writerow(data)

    def write_average_training_time(self, blocks, timers):
        with open(log_service.build_path('csv', 'avg_training_time.csv'), mode='a+', newline='') as f:
            writer = csv.writer(f)
            avg_time = statistics.mean(timers)
            writer.writerow([blocks, avg_time])

    def write_sliding_blocks_training_time(self, current_blocks, timer, listed_space,
                                            state_space, reindex_function):
        '''
        Write on csv the training time, that will be used for regressor training.
        Use sliding blocks mechanism to multiple the entries.

        Args:
            current_blocks ([type]): [description]
            timer ([type]): [description]
            listed_space ([type]): [description]
            state_space ([type]): [description]
            reindex_function ([type]): [description]
        '''

        with open(log_service.build_path('csv', 'training_time.csv'), mode='a+', newline='') as f:
            writer = csv.writer(f)

            # generate all possible 'sliding blocks' combinations
            for i in range(current_blocks, self.blocks + 1):
                data = [timer, current_blocks]

                # slide block forward
                for _ in range(current_blocks, i):
                    data.extend([0, 0, 0, 0])

                # add reindexed block encoding
                encoded_child = state_space.entity_encode_child(listed_space)
                concatenated_child = np.concatenate(encoded_child, axis=None).astype('int32')
                reindexed_child = []
                for j, action_index in enumerate(concatenated_child):
                    if j % 2 == 0:
                        # TODO: investigate this
                        reindexed_child.append(action_index + 1)
                    else:
                        reindexed_child.append(reindex_function(action_index))
                data.extend(reindexed_child)

                # extend with empty blocks, if necessary
                for _ in range(i+1, self.blocks + 1):
                    data.extend([0, 0, 0, 0])

                writer.writerow(data)

    def process(self):
        '''
        Main function, executed by run.py to start POPNAS algorithm.
        '''

        # create the complete headers row of the CSV files
        headers = ["time", "blocks"]
        op_timers = []
        t_max = 0

        if self.restore:
            starting_B = self.checkpoint  # change the starting point of B

            self._logger.info("Loading operator indeces!")
            with open(log_service.build_path('csv', 'timers.csv')) as f:
                reader = csv.reader(f)
                for row in reader:
                    elem = float(row[0])
                    op_timers.append(elem)
                    if elem >= t_max:
                        t_max = elem
            self._logger.info(op_timers, t_max)

        else:
            starting_B = 0

            # create headers for csv files
            for b in range(1, self.blocks+1):
                a = b*2
                c = a-1
                new_block = [f"input_{c}", f"operation_{c}", f"input_{a}", f"operation_{a}"]
                headers.extend(new_block)

            # add headers to training_time.csv
            with open(log_service.build_path('csv', 'training_time.csv'), mode='a+', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)

        # TODO: restore search space
        operators = ['identity', '3x3 dconv', '5x5 dconv', '7x7 dconv', '1x7-7x1 conv', '3x3 conv', '3x3 maxpool', '3x3 avgpool']
        #operators = ['identity', '3x3 dconv']

        # construct a state space
        state_space = StateSpace(self.blocks, input_lookback_depth=-2, input_lookforward_depth=None, operators=operators)

        # print the state space being searched
        state_space.print_state_space()

        # load correct dataset (based on self.dataset)
        (x_train_init, y_train_init), (x_test_init, y_test_init) = self.load_dataset()

        dataset = self.prepare_dataset(x_train_init, y_train_init, x_test_init, y_test_init)

        # create the ControllerManager and build the internal policy network
        controller = ControllerManager(state_space, self.checkpoint, B=self.blocks, K=self.children,
                                       train_iterations=15,
                                       pnas_mode=self.pnas_mode,
                                       restore_controller=self.restore)

        # create the Network Manager
        manager = NetworkManager(dataset, data_num=self.sets, epochs=self.epochs, batchsize=self.batchsize,
                                 learning_rate=self.learning_rate, filters=self.filters)

        # if B = 0, perform initial thrust before starting actual training procedure
        if starting_B == 0:
            k = None
            self.perform_initial_thrust(state_space, manager)
            
            starting_B = 1
        else:
            k = self.children

        monoblock_times = []
        reindex_function = None

        # train the child CNN networks for each number of blocks
        for current_blocks in range(starting_B, self.blocks + 1):
            actions = controller.get_actions(top_k=k)  # get all actions for the previous state
            rewards = []
            timers = []

            for t, action in enumerate(actions):
                reward, timer, listed_space = self.generate_and_train_model_from_actions(state_space, manager, action, t+1, len(actions))
                rewards.append(reward)
                timers.append(timer)

                if current_blocks == 1:
                    monoblock_times.append([timer, listed_space])

                    # get required data for dynamic reindex
                    # op_timers will contain timers for blocks with both same operation, for each operation, in order
                    if (listed_space[0] == listed_space[2] and listed_space[1] == listed_space[3] and listed_space[0] == -1):
                        with open(log_service.build_path('csv', 'timers.csv'), mode='a+', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([timer])
                            op_timers.append(timer)

                            if timer >= t_max:
                                t_max = timer

                self._logger.info("Finished %d out of %d models!", (t + 1), len(actions))

                # write the results of this trial into a file
                with open(log_service.build_path('csv', 'real_accuracy.csv'), mode='a+', newline='') as f:
                    data = [reward, current_blocks]
                    data.extend(listed_space)
                    writer = csv.writer(f)
                    writer.writerow(data)

                # in current_blocks = 1 case, we need all CNN to be able to dynamic reindex
                if current_blocks > 1:
                    self.write_sliding_blocks_training_time(current_blocks, timer, listed_space,
                                                            state_space, reindex_function)

            # current_blocks = 1 case, same mechanism but wait all CNN for applying dynamic reindex
            if current_blocks == 1:
                reindex_function = self.generate_dynamic_reindex_function(operators, op_timers, t_max)
                for timer, listed_space in monoblock_times:
                    self.write_sliding_blocks_training_time(current_blocks, timer, listed_space,
                                                            state_space, reindex_function)
            
            self.write_average_training_time(current_blocks, timers)

            loss = controller.train_step(rewards)
            self._logger.info("Trial %d: ControllerManager loss : %0.6f", current_blocks, loss)

            controller.update_step(headers, reindex_function)

        self._logger.info("Finished!")
