from types import prepare_class
import numpy as np
import csv
from sklearn.model_selection import train_test_split

import importlib.util

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # disable GPU debugging info

import tensorflow as tf

from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.datasets import cifar100
from tensorflow.python.keras.utils import to_categorical

from encoder import StateSpace
from controller import ControllerManager
from manager import NetworkManager
from model import ModelGenerator

import log_service

# should make only CPU visible
#os.environ['CUDA_VISIBLE_DEVICES'] = ''    

# if tf.test.gpu_device_name():
#     _logger.info('GPU found')
# else:
#     _logger.info("No GPU found")

# device_list = tf.config.experimental.get_visible_devices()
# _logger.info(device_list)


class Train:

    def __init__(self, blocks, children, checkpoint,
                 dataset, sets, epochs, batchsize,
                 learning_rate, restore, cpu):

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
        self.cpu = cpu


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
        for i in range(0, self.sets):
            # create a validation set for evaluation of the child models
            x_train, x_validation, y_train, y_validation = train_test_split(x_train_init, y_train_init, test_size=0.1, random_state=0)
            
            # TODO: take only 400 images for really fast training, delete this in future
            x_train = x_train[:400]
            y_train = y_train[:400]

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


    def process(self):
    
        # create the complete headers row of the CSV files
        headers = ["time", "blocks"]
        index_list = [0]
        t_max = 0

        if self.restore == True:
            starting_B = self.checkpoint - 1 # change the starting point of B

            self._logger.info("Loading operator indeces!")
            with open(log_service.build_path('csv', 'timers.csv')) as f:
                reader = csv.reader(f)
                for row in reader:
                    elem = float(row[0])
                    index_list.append(elem)
                    if elem >= t_max :
                        t_max = elem
            self._logger.info(index_list, t_max)
                         
        else:
            starting_B = 0

            # create headers for csv files
            for b in range(1, self.blocks+1):
                a = b*2
                c = a-1
                new_block = ["input_%d" % c , "operation_%d" % c, "input_%d" % a, "operation_%d" % a]
                headers.extend(new_block)
            
            # add headers to training_time.csv
            with open(log_service.build_path('csv', 'training_time.csv'), mode='a+', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)

        # TODO: restore search space
        #operators = ['identity', '3x3 dconv', '5x5 dconv', '7x7 dconv', '1x7-7x1 conv', '3x3 conv', '3x3 maxpool', '3x3 avgpool']
        operators = ['identity']

        # construct a state space
        state_space = StateSpace(self.blocks, input_lookback_depth=-2, input_lookforward_depth=None, operators=operators)

        # print the state space being searched
        state_space.print_state_space()
        NUM_TRAILS = state_space.print_total_models(self.children)

        # load correct dataset (based on self.dataset)   
        (x_train_init, y_train_init), (x_test_init, y_test_init) = self.load_dataset()
            
        dataset = self.prepare_dataset(x_train_init, y_train_init, x_test_init, y_test_init)


        # create the ControllerManager and build the internal policy network
        controller = ControllerManager(state_space, self.checkpoint, B=self.blocks, K=self.children,
                                       train_iterations=10,
                                       reg_param=0,
                                       controller_cells=100,
                                       restore_controller=self.restore)

        # create the Network Manager
        manager = NetworkManager(dataset, data_num=self.sets, epochs=self.epochs, batchsize=self.batchsize,
                                 learning_rate=self.learning_rate, cpu=self.cpu)

        block_times = []
            
        # train for number of trials
        for trial in range(starting_B, self.blocks):
            if trial == 0:
                k = None
                # build a starting point model with 0 blocks to evaluate the offset
                self._logger.info("Model #1 / #1")
                self._logger.info("\t%s", state_space.parse_state_space_list([]))
                reward, timer = manager.get_rewards(ModelGenerator, state_space.parse_state_space_list([]))
                self._logger.info("Final Accuracy: %0.6f", reward)
                self._logger.info("Training time: %0.6f", timer)
                with open(log_service.build_path('csv', 'training_time.csv'), mode='a+', newline='') as f:
                    data = [timer, 0]
                    for i in range(self.blocks):
                        data.extend([0, 0, 0, 0])
                    writer = csv.writer(f)
                    writer.writerow(data)           
            else:
                k = self.children
                
            actions = controller.get_actions(top_k=k)  # get all actions for the previous state
            rewards = []
            timers = []
            
            for t, action in enumerate(actions):
                # print the action probabilities
                state_space.print_actions(action)
                listed_space = state_space.parse_state_space_list(action)
                self._logger.info("Model #%d / #%d" % (t + 1, len(actions)))
                self._logger.info("\t%s", listed_space)

                # build a model, train and get reward and accuracy from the network manager
                reward, timer = manager.get_rewards(ModelGenerator, listed_space)
                self._logger.info("Final Accuracy : %0.6f", reward)
                self._logger.info("Training time : %0.6f", timer)

                rewards.append(reward)
                timers.append(timer)
                if trial == 0 :
                    block_times.append([timer, listed_space])
                    if (listed_space[0] == listed_space[2] and listed_space[1] == listed_space[3] and listed_space[0] == -1) :
                        with open(log_service.build_path('csv', 'timers.csv'), mode='a+', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([timer])
                            index_list.append(timer)
                            if timer >= t_max:
                                t_max = timer
                self._logger.info("Finished %d out of %d models!" % (t + 1, len(actions)))

                # write the results of this trial into a file
                with open(log_service.build_path('csv', 'real_accuracy.csv'), mode='a+', newline='') as f:
                    data = [reward, trial+1]
                    data.extend(listed_space)
                    writer = csv.writer(f)
                    writer.writerow(data)

                if trial > 0 :
                    # write the forward pass time of this trial into a file
                    with open(log_service.build_path('csv', 'training_time.csv'), mode='a+', newline='') as f:
                        writer = csv.writer(f)
                        for i in range(trial, self.blocks):
                            data = [timer, trial+1]
                            for j in range(trial, i):
                                data.extend([0, 0, 0, 0])
                            encoded_child = state_space.entity_encode_child(listed_space) #
                            concatenated_child = np.concatenate(encoded_child, axis=None).astype('int32') #
                            reindexed_child = []
                            for index, elem in enumerate(concatenated_child):
                                elem = elem + 1
                                if index % 2 == 0:
                                    reindexed_child.append(elem)
                                else:
                                    reindexed_child.append(len(operators) * index_list[elem] / t_max)
                            data.extend(reindexed_child)
                            for j in range(i+1, self.blocks):
                                data.extend([0, 0, 0, 0])
                            writer.writerow(data)

            if trial == 0:
                with open(log_service.build_path('csv', 'training_time.csv'), mode='a+', newline='') as f:
                    writer = csv.writer(f)
                    for row in block_times:
                        for i in range(trial, self.blocks):
                            data = [row[0], trial+1]
                            for j in range(trial, i):
                                data.extend([0, 0, 0, 0])
                            encoded_child = state_space.entity_encode_child(row[1]) #
                            concatenated_child = np.concatenate(encoded_child, axis=None).astype('int32') #
                            reindexed_child = []
                            for index, elem in enumerate(concatenated_child):
                                elem = elem + 1
                                if index % 2 == 0:
                                    reindexed_child.append(elem)
                                else:
                                    reindexed_child.append(len(operators) * index_list[elem] / t_max)
                            data.extend(reindexed_child)
                            for j in range(i+1, self.blocks):
                                data.extend([0, 0, 0, 0])
                            writer.writerow(data)    
            
            loss = controller.train_step(rewards)
            self._logger.info("Trial %d: ControllerManager loss : %0.6f" % (trial + 1, loss))

            controller.update_step(headers, t_max, len(operators), index_list, timers, lookback=-2)
            
        self._logger.info("Finished!")
