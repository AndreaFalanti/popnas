import numpy as np
import csv
from sklearn.model_selection import train_test_split
# from sklearn.cross_validation import train_test_split

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import importlib.util

import inspect

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # disable GPU debugging info

import configparser

import time

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical

from encoder import ControllerManager, StateSpace
from manager import NetworkManager
from model import ModelGenerator

import sys
import argparse
import pandas


class Train:

    def __init__(self, blocks, children, checkpoint,
                 dataset, sets, epochs, batchsize,
                 learning_rate, restore, timestr):

        self.blocks = blocks
        self.checkpoint = checkpoint
        self.children = children
        self.dataset = dataset
        self.sets = sets
        self.epochs = epochs
        self.batchsize = batchsize
        self.learning_rate = learning_rate
        self.restore = restore
        self.timestr = timestr

    def process(self):
        # I don't know why I am doing this =(
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
        # create the complete headers row of the CSV files
        headers = ["time", "blocks"]
        index_list = [0]
        t_max = 0

        # create headers for csv files
        for b in range(1, self.blocks + 1):
            a = b * 2
            c = a - 1
            new_block = ["input_%d" % c, "operation_%d" % c, "input_%d" % a, "operation_%d" % a]
            headers.extend(new_block)

        if self.restore == True:
            timestr = self.timestr
            starting_B = self.checkpoint - 1  # change the starting point of B

            print("Loading operator indeces !")
            with open('logs/%s/csv/timers.csv' % timestr) as f:
                reader = csv.reader(f)
                for row in reader:
                    elem = float(row[0])
                    index_list.append(elem)
                    if elem >= t_max:
                        t_max = elem
            print(index_list, t_max)

        else:
            timestr = time.strftime('%Y-%m-%d-%H-%M-%S')  # get time for logs folder
            os.makedirs('logs/%s/csv' % timestr)  # create .csv path
            os.mkdir('logs/%s/ini' % timestr)  # create .ini folder
            starting_B = 0

            # add headers to training_time.csv
            with open('logs/%s/csv/training_time.csv' % timestr, mode='a+', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)

        # initialize use_columns string for the configuration file
        use_columns = '\"blocks\",'

        # operators = ['identity', '3x3 dconv', '5x5 dconv', '7x7 dconv', '1x7-7x1 conv', '3x3 conv', '3x3 maxpool', '3x3 avgpool']
        operators = ['identity', '3x3 conv']

        # construct a state space
        state_space = StateSpace(self.blocks, input_lookback_depth=-2, input_lookforward_depth=None, operators=operators)

        # print the state space being searched
        state_space.print_state_space()
        NUM_TRAILS = state_space.print_total_models(self.children)

        # reduce the dataset dimension
        '''
        cifar = inspect.getsource(cifar10)
        new_cifar = cifar.replace('50000', '10000')
        new_new_cifar = new_cifar.replace('range(1, 6)', 'range(1, 2)')
        exec(new_new_cifar, cifar10.__dict__)
        '''

        if self.dataset == "cifar10":
            (x_train_init, y_train_init), (x_test_init, y_test_init) = cifar10.load_data()

        elif self.dataset == "cifar100":
            (x_train_init, y_train_init), (x_test_init, y_test_init) = cifar100.load_data()

        else:
            spec = importlib.util.spec_from_file_location("dataset", self.dataset)
            set = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(set)
            (x_train_init, y_train_init), (x_test_init, y_test_init) = set.load_data()

        # reduce dataset dimensions to 10000 samples
        x_train_init = x_train_init[:10000]
        y_train_init = y_train_init[:10000]

        x_train_init = x_train_init.astype('float32') / 255.
        x_test_init = x_test_init.astype('float32') / 255.

        dataset = []
        for i in range(self.sets):
            # create a validation set for evaluation of the child models
            x_train, x_test, y_train, y_test = train_test_split(x_train_init, y_train_init, test_size=0.1, random_state=0)

            if self.dataset == "cifar10":
                # cifar10
                y_train = to_categorical(y_train, 10)
                y_test = to_categorical(y_test, 10)

            elif self.dataset == "cifar100":
                # cifar100
                y_train = to_categorical(y_train, 100)
                y_test = to_categorical(y_test, 100)

            dataset.append([x_train, y_train, x_test, y_test])  # pack the dataset for the NetworkManager

        # create the ControllerManager and build the internal policy network
        controller = ControllerManager(state_space, timestr, self.checkpoint, B=self.blocks, K=self.children,
                                       train_iterations=10,
                                       reg_param=0,
                                       controller_cells=100,
                                       restore_controller=self.restore)

        # create the Network Manager
        manager = NetworkManager(dataset, timestr, data_num=self.sets, epochs=self.epochs, batchsize=self.batchsize, learning_rate=self.learning_rate)
        print()

        block_times = []

        # train for number of trials
        for trial in range(starting_B, self.blocks):
            if trial == 0:
                k = None
                # build a starting point model with 0 blocks to evaluate the offset
                print("Model #1 / #1")
                print(" ", state_space.parse_state_space_list([]))
                reward, timer = manager.get_rewards(ModelGenerator, state_space.parse_state_space_list([]), timestr)
                print("Final Accuracy : ", reward)
                print("Training time : ", timer)
                with open('logs/%s/csv/training_time.csv' % timestr, mode='a+', newline='') as f:
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
            encoded_actions = []

            for t, action in enumerate(actions):
                # print the action probabilities
                state_space.print_actions(action)
                listed_space = state_space.parse_state_space_list(action)
                print("Model #%d / #%d" % (t + 1, len(actions)))
                print(" ", listed_space)

                # build a model, train and get reward and accuracy from the network manager
                reward, timer = manager.get_rewards(ModelGenerator, listed_space)
                print("Final Accuracy : ", reward)
                print("Training time : ", timer)

                rewards.append(reward)
                timers.append(timer)
                if trial == 0:
                    block_times.append([timer, listed_space])
                    if (listed_space[0] == listed_space[2] and listed_space[1] == listed_space[3] and listed_space[0] == -1):
                        with open('logs/%s/csv/timers.csv' % timestr, mode='a+', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([timer])
                            index_list.append(timer)
                            if timer >= t_max:
                                t_max = timer
                print("\nFinished %d out of %d models ! \n" % (t + 1, len(actions)))

                # write the results of this trial into a file
                with open('logs/%s/csv/real_accuracy.csv' % timestr, mode='a+', newline='') as f:
                    data = [reward, trial + 1]
                    data.extend(listed_space)
                    writer = csv.writer(f)
                    writer.writerow(data)

                if trial > 0:
                    # write the forward pass time of this trial into a file
                    with open('logs/%s/csv/training_time.csv' % timestr, mode='a+', newline='') as f:
                        writer = csv.writer(f)
                        for i in range(trial, self.blocks):
                            data = [timer, trial + 1]
                            for j in range(trial, i):
                                data.extend([0, 0, 0, 0])
                            encoded_child = state_space.entity_encode_child(listed_space)  #
                            concatenated_child = np.concatenate(encoded_child, axis=None).astype('int32')  #
                            reindexed_child = []
                            for index, elem in enumerate(concatenated_child):
                                elem = elem + 1
                                if index % 2 == 0:
                                    reindexed_child.append(elem)
                                else:
                                    reindexed_child.append(len(operators) * index_list[elem] / t_max)
                            data.extend(reindexed_child)
                            for j in range(i + 1, self.blocks):
                                data.extend([0, 0, 0, 0])
                            writer.writerow(data)

            if trial == 0:
                with open('logs/%s/csv/training_time.csv' % timestr, mode='a+', newline='') as f:
                    writer = csv.writer(f)
                    for row in block_times:
                        for i in range(trial, self.blocks):
                            data = [row[0], trial + 1]
                            for j in range(trial, i):
                                data.extend([0, 0, 0, 0])
                            encoded_child = state_space.entity_encode_child(row[1])  #
                            concatenated_child = np.concatenate(encoded_child, axis=None).astype('int32')  #
                            reindexed_child = []
                            for index, elem in enumerate(concatenated_child):
                                elem = elem + 1
                                if index % 2 == 0:
                                    reindexed_child.append(elem)
                                else:
                                    reindexed_child.append(len(operators) * index_list[elem] / t_max)
                            data.extend(reindexed_child)
                            for j in range(i + 1, self.blocks):
                                data.extend([0, 0, 0, 0])
                            writer.writerow(data)

            loss = controller.train_step(rewards)
            print("Trial %d: ControllerManager loss : %0.6f" % (trial + 1, loss))

            controller.update_step(headers, t_max, len(operators), index_list, timers, lookback=-2)
            print()

        print("Finished !")

