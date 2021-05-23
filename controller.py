import pandas
import numpy as np
import csv

import tensorflow as tf

import os
import configparser

from encoder import StateSpace
from aMLLibrary import sequence_data_processing

import log_service
_logger = log_service.getLogger(__name__)


class Controller(tf.keras.Model):
    
    def __init__(self, controller_cells, embedding_dim,
                 input_embedding_max, operator_embedding_max):
        '''
        LSTM Controller model which accepts encoded sequence describing the
        architecture of the model and predicts a singular value describing
        its probably validation accuracy.
        
        # Args:
            controller_cells: number of cells of the Controller LSTM.
            embedding_dim: size of the embedding dimension.
            input_embedding_max: maximum input dimension of the input embedding.
            operator_embedding_max: maximum input dimension of the operator encoding.
        '''
        super(Controller, self).__init__(name='EncoderRNN')
        self.controller_cells = controller_cells
        self.embedding_dim = embedding_dim
        self.input_embedding_max = input_embedding_max
        self.operator_embedding_max = operator_embedding_max

        # Layers
        self.input_embedding = tf.keras.layers.Embedding(input_embedding_max + 1, embedding_dim)
        self.operators_embedding = tf.keras.layers.Embedding(operator_embedding_max + 1, embedding_dim)
        '''  
        if tf.test.is_gpu_available():
            self.rnn = tf.keras.layers.CuDNNLSTM(controller_cells, return_state=True)
        else:
            self.rnn = tf.keras.layers.LSTM(controller_cells, return_state=True)
        '''
        self.rnn = tf.keras.layers.LSTM(controller_cells, return_state=True)
    
        self.rnn_score = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs_operators, states=None, training=None, mask=None):
        inputs, operators = self._get_inputs_and_operators(inputs_operators)  # extract the data       
        if states is None:  # initialize the state vectors
            states = self.rnn.get_initial_state(inputs)
            states = [tf.to_float(state) for state in states]
        
        # map the sparse inputs and operators into dense embeddings
        embed_inputs = self.input_embedding(inputs)
        embed_ops = self.operators_embedding(operators)
        
        # concatenate the embeddings
        embed = tf.concat([embed_inputs, embed_ops], axis=-1)  # concatenate the embeddings
        
        # run over the LSTM
        out = self.rnn(embed, initial_state=states)
        out, h, c = out  # unpack the outputs and states
        
        # get the predicted validation accuracy
        score = self.rnn_score(out)

        return [score, [h, c]]

    def _get_inputs_and_operators(self, inputs_operators):
        '''
        Splits the joint inputs and operators into seperate inputs
        and operators list for convenience of the SearchSpace.
        
        # Args:
            inputs_operators: interleaved [input; operator] pairs.

        # Returns:
            list of inputs and list of operators.
        '''
        inputs = inputs_operators[:, 0::2]  # even place data
        operators = inputs_operators[:, 1::2]  # odd place data

        return inputs, operators


class ControllerManager:
    '''
    Utility class to manage the RNN Controller.
    
    Tasked with maintaining the state of the training schedule,
    keep track of the children models generated from cross-products,
    cull non-optimal children model configurations and resume
    training.
    '''
    def __init__(self, state_space, timestr,
                 checkpoint_B,
                 B=5, K=256, T=np.inf,
                 train_iterations=10,
                 reg_param=0.001,
                 controller_cells=32,
                 embedding_dim=20,
                 input_B=None,
                 restore_controller=False):
        '''
        Manages the Controller network training and prediction process.
        
        # Args:
            state_space: completely defined search space.
            timestr: time string to create the log folder.
            B: depth of progression.
            K: maximum number of children model trained per level of depth.
            T: maximum training time.
            train_iterations: number of training epochs for the RNN per depth level.
            reg_param: strength of the L2 regularization loss.
            controller_cells: number of cells in the Controller LSTM.
            embedding_dim: embedding dimension for inputs and operators.
            input_B: override value of B, used only when we are restoring the controller.
                Determing the maximum input connectivity allowed to the RNN Controller,
                to maintain backward compatibility with trained models.
                
                Use it alongside `restore_controller` to evaluate model settings
                with larger depth `B` than allowed at training time.
            restore_controller: flag whether to restore a pre-trained RNN controller
                upon construction.
        '''
        
        self.state_space = state_space  # type: StateSpace
        self.state_size = self.state_space.size
        self.timestr = timestr

        self.global_epoch = 0

        self.B = B
        self.K = K
        self.T = T
        self.embedding_dim = embedding_dim

        self.train_iterations = train_iterations
        self.controller_cells = controller_cells
        self.reg_strength = reg_param
        self.input_B = input_B
        self.restore_controller = restore_controller

        # restore controller
        if self.restore_controller == True:
            self.b_ = checkpoint_B
            _logger.info("Loading controller history !")

            next_children = []
            
            #read next_children from .csv file
            with open('logs/%s/csv/next_children.csv' % self.timestr, newline='') as f:
                reader = csv.reader(f, delimiter = ',')
                for row in reader:
                    encoded_row = []
                    for i in range(len(row)):
                        if i % 2 == 0:
                            encoded_row.append(int(row[i]))
                        else:
                            encoded_row.append(row[i])
                    next_children.append(encoded_row)

            for i in range(1, self.b_):
                #read children from .csv file
                with open('logs/%s/csv/children_%s.csv' % (self.timestr, i), newline='') as f:
                    reader = csv.reader(f, delimiter = ',')
                    j = 0
                    for row in reader:
                        encoded_row = []
                        for el in range(len(row)):
                            if el % 2 == 0:
                                encoded_row.append(int(row[el]))
                            else:
                                encoded_row.append(row[el])
                        np_encoded_row = np.array(encoded_row, dtype=np.object)
                        if j == 0 :
                            children_i = [np_encoded_row]
                        else :
                            children_i = np.concatenate((children_i, [np_encoded_row]), axis=0)
                        j = j + 1

                # read old rewards from .csv file
                with open('logs/%s/csv/rewards_%s.csv' % (self.timestr, i), newline='') as f:
                    reader = csv.reader(f, delimiter = ',')
                    j = 0
                    for row in reader:
                        if j == 0 :
                            rewards_i = [float(row[0])]
                        else :
                            rewards_i.append(float(row[0]))
                        j = j + 1
                    rewards_i = np.array(rewards_i, dtype=np.float32)

                if i == 1 :
                    children = [children_i]
                    rewards = [rewards_i]
                else :
                    children.append(children_i)
                    rewards.append(rewards_i)
            
            self.state_space.update_children(next_children)
            self.children_history = children

            self.score_history = rewards
            
        else :
            self.b_ = 1
            self.children_history = None
            self.score_history = None

        self.build_policy_network()

    def get_actions(self, top_k=None):
        '''
        Gets a one hot encoded action list, either from random sampling or from
        the ControllerManager RNN

        # Args:
            top_k: Number of models to return

        # Returns:
            A one hot encoded action list
        '''
        models = self.state_space.children

        if top_k is not None:
            models = models[:top_k]

        actions = []
        for model in models:
            encoded_model = self.state_space.entity_encode_child(model)
            actions.append(encoded_model)

        return actions

    def build_policy_network(self):
        '''
        Construct the RNN controller network with the provided settings.
        
        Also constructs saver and restorer to the RNN controller if required.
        '''
        '''
        if tf.test.is_gpu_available():
            device = '/gpu:0'
        else:
            device = '/cpu:0'
        '''    
        device = '/cpu:0'
        self.device = device

        if self.restore_controller and self.input_B is not None:
            input_B = self.input_B
        else:
            input_B = self.state_space.inputs_embedding_max

        self.global_step = tf.compat.v1.train.get_or_create_global_step()
        learning_rate = tf.compat.v1.train.exponential_decay(0.001, self.global_step, 500, 0.98, staircase=True)

        with tf.device(device):
            self.controller = Controller(self.controller_cells,
                                         self.embedding_dim,
                                         input_B,
                                         self.state_space.operator_embedding_max)

            self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

        self.saver = tf.train.Checkpoint(controller=self.controller,
                                         optimizer=self.optimizer,
                                         global_step=self.global_step)

        if self.restore_controller:
            path = tf.train.latest_checkpoint('logs/%s/weights' % self.timestr)

            if path is not None and tf.train.checkpoint_exists(path):
                _logger.info("Loading controller checkpoint!")
                self.saver.restore(path)

    def loss(self, real_acc, rnn_scores):
        '''
        Computes the surrogate losses to train the controller.
        
        - rnn score loss is the MSE between the real validation acc and the
        predicted acc of the rnn.
        
        - reg loss is the L2 regularization loss on the parameters of the controller.
        
        # Args:
            real_acc: actual validation accuracy obtained by child models.
            rnn_scores: predicted validation accuracy obtained by child models.

        # Returns:
            weighted sum of rnn score loss + reg loss.
        '''
        # RNN predicted accuracies
        rnn_score_loss = tf.losses.mean_squared_error(real_acc, rnn_scores)

        # Regularization of model
        params = self.controller.trainable_variables
        reg_loss = tf.reduce_sum([tf.nn.l2_loss(x) for x in params])

        total_loss = rnn_score_loss + self.reg_strength * reg_loss

        return total_loss

    def train_step(self, rewards):
        '''
        Perform a single train step on the Controller RNN

        # Returns:
            final training loss
        '''
        children = np.array(self.state_space.children, dtype=np.object)  # take all the children
        rewards = np.array(rewards, dtype=np.float32)
        _logger.info("Rewards : %s", rewards)
        loss = 0

        if self.children_history is None:
            self.children_history = [children]
            self.score_history = [rewards]
            batchsize = rewards.shape[0]
        else:
            self.children_history.append(children)
            self.score_history.append(rewards)
            batchsize = sum([data.shape[0] for data in self.score_history])
            
        train_size = batchsize * self.train_iterations
        _logger.info("Controller: Number of training steps required for this stage : %d", train_size)

        #logs
        self.logdir = 'logs/%s/controller' % self.timestr
        summary_writer = tf.contrib.summary.create_file_writer(self.logdir)
        summary_writer.set_as_default()

        for current_epoch in range(self.train_iterations):
            _logger.info("Controller: Begin training epoch %d", current_epoch + 1)

            self.global_epoch = self.global_epoch + 1

            for dataset_id in range(self.b_):
                children = self.children_history[dataset_id]
                scores = self.score_history[dataset_id]
                ids = np.array(list(range(len(scores))))
                np.random.shuffle(ids)

                _logger.info("Controller: Begin training - B = %d", dataset_id + 1)

                for id, (child, score) in enumerate(zip(children[ids], scores[ids])):
                    child = child.tolist()
                    state_list = self.state_space.entity_encode_child(child)
                    state_list = np.concatenate(state_list, axis=-1).astype('int32')
                    
                    with tf.device(self.device):
                        state_list = tf.convert_to_tensor(state_list)

                        with tf.GradientTape() as tape:
                            rnn_scores, states = self.controller(state_list, states=None)
                            acc_scores = score.reshape((1, 1))

                            total_loss = self.loss(acc_scores, rnn_scores)

                        grads = tape.gradient(total_loss, self.controller.trainable_variables)
                        grad_vars = zip(grads, self.controller.trainable_variables)

                        self.optimizer.apply_gradients(grad_vars, self.global_step)

                    loss += total_loss.numpy().sum()

                _logger.info("Controller: Finished training epoch %d / %d of B = %d / %d", current_epoch + 1, self.train_iterations, dataset_id + 1, self.b_)

            # add accuracy to Tensorboard
            with tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar("average_accuracy", rewards.mean(), family="controller", step=self.global_epoch)
                tf.contrib.summary.scalar("average_loss", loss.mean(), family="controller", step=self.global_epoch)

        with open('logs/%s/csv/rewards.csv' % self.timestr, mode='a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(map(lambda x: [x], rewards))
            
        # save weights
        if not os.path.exists('logs/%s/weights' % self.timestr):
            os.makedirs('logs/%s/weights' % self.timestr) # create weights folder
        self.saver.save('logs/%s/weights/controller.ckpt' % self.timestr)

        return loss.mean()

    def update_step(self, headers, t_max, op_size, index_list, timers, lookback=0):
        '''
        Updates the children from the intermediate products for the next generation
        of larger number of blocks in each cell
        '''
        
        # plot controller architecture
        # plot_model(self.controller, to_file='%s/controller_plot.png' % self.logdir, show_shapes=True, show_layer_names=True)
        
        if self.b_ == 2:
            df = pandas.read_csv('logs/%s/csv/training_time.csv' % self.timestr, skiprows=[1])
        else:
            df = pandas.read_csv('logs/%s/csv/training_time.csv' % self.timestr)
        
        df.to_csv('logs/%s/csv/training_time.csv' % self.timestr, na_rep=0, index=False)

        inputs = []
        for b in range(1, (self.B+1)):
            a = b*2
            c = a-1
            new_block = ["input_%d" % c , "input_%d" % a]
            inputs.extend(new_block)

        # create the NNLS configuration file
        config = configparser.ConfigParser()
        config['General'] = {'run_num' : 1,
                             'techniques' : ['NNLS'],
                             'hp_selection' : 'All',
                             'validation' : 'All',
                             'y' : '"time"',
                             'generate_plots' : 'True'}
        config['DataPreparation'] = {'input_path' : 'logs/%s/csv/training_time.csv' % self.timestr,
                                     'skip_columns' : inputs}
        config['NNLS'] = {'fit_intercept' : [True, False]}
    
        with open('logs/%s/ini/training_time_NNLS_%d.ini' % (self.timestr, self.b_), 'w') as f:
            config.write(f)
        
        # a-MLLibrary
        sequence_data_processor_NNLS = sequence_data_processing.SequenceDataProcessing('logs/%s/ini/training_time_NNLS_%d.ini' % (self.timestr, self.b_), output='logs/%s/output_NNLS_%d' % (self.timestr, self.b_))
        regressor_NNLS = sequence_data_processor_NNLS.process()
        if self.b_ + 1 <= self.B:
            self.b_ += 1
            models_scores = []
         
            # iterate through all the intermediate children
            for i, intermediate_child in enumerate(self.state_space.prepare_intermediate_children(self.b_)):
                state_list = self.state_space.entity_encode_child(intermediate_child)
                state_list = np.concatenate(state_list, axis=-1).astype('int32')
                state_list = tf.convert_to_tensor(state_list)

                # save predicted times on a .csv file
                with open('logs/%s/csv/predicted_time_%d.csv' % (self.timestr, self.b_), mode='a+', newline='') as f:
                    writer = csv.writer(f)
                    encoded_child = self.state_space.entity_encode_child(intermediate_child)
                    concatenated_child = np.concatenate(encoded_child, axis=None).astype('int32')
                    reindexed_child = []
                    for index, elem in enumerate(concatenated_child):
                        elem = elem + 1
                        if index % 2 == 0:
                            reindexed_child.append(elem)
                        else:
                            reindexed_child.append(op_size * index_list[elem] / t_max)
                    reindexed_child = np.concatenate(reindexed_child, axis=None).astype('int32')
                    array = np.append(np.array([0, self.b_]), [x for x in reindexed_child])
                    for b in range(self.b_, self.B):
                        array = np.append(array, np.array([0, 0, 0, 0]))
                    df_row = pandas.DataFrame([array], columns=headers)
                    data = [regressor_NNLS.predict(df_row)[0]]
                    data.extend(intermediate_child)
                    writer.writerow(data)
                
                # score the child
                score, _ = self.controller(state_list, states=None)
                score = score[0, 0].numpy()

                # preserve the child and its score
                if data[0] <= self.T:
                    models_scores.append([intermediate_child, score, regressor_NNLS.predict(df_row)[0]])

                # save predicted scores on a .csv file
                with open('logs/%s/csv/predicted_accuracy_%d.csv' % (self.timestr, self.b_), mode='a+', newline='') as f:
                    writer = csv.writer(f)
                    data = [score]
                    data.extend(intermediate_child)
                    writer.writerow(data)

                if (i + 1) % 500 == 0:
                    _logger.info("Scored %d models. Current model score = %0.4f", i + 1, score)

            # sort the children according to their score
            models_scores = sorted(models_scores, key=lambda x: x[1], reverse=True)

            pareto_front = [models_scores[0]]
            for pair in models_scores[1:] :
                if pair[2] <= pareto_front[-1][2] :
                    pareto_front.append(pair)
            for row in pareto_front :
                with open('logs/%s/csv/pareto_front_%d.csv' % (self.timestr, self.b_), mode='a+', newline='') as f:
                    writer = csv.writer(f)
                    data = [row[2], row[1]]
                    data.extend(row[0])
                    writer.writerow(data)

            # account for case where there are fewer children than K
            if self.K is not None:
                children_count = min(self.K, len(pareto_front))
            else:
                children_count = len(pareto_front)

            # take only the K highest scoring children for next iteration
            children = []
            for i in range(children_count):
                children.append(pareto_front[i][0])
                with open('logs/%s/csv/children.csv' % self.timestr, mode='a+', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(pareto_front[i][0])

            # save these children for next round
            self.state_space.update_children(children)
        else:
            _logger.info("No more updates necessary as max B has been reached!")
