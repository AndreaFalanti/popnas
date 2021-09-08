import pandas
import numpy as np
import csv
from tqdm import tqdm

import tensorflow as tf

from configparser import ConfigParser
import os
import sys

from encoder import StateSpace, CellEncoding
from aMLLibrary import sequence_data_processing

import log_service
from utils.stream_to_logger import StreamToLogger
from contextlib import redirect_stderr, redirect_stdout


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

        # Tensorflow2 now automatically use CuDNNLSTM if using GPU and LSTM has the right parameters (default ones are good)
        # check this url for more info: https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
        self.rnn = tf.keras.layers.LSTM(controller_cells, return_state=True)

        self.rnn_score = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs_operators, states=None, training=None, mask=None):
        inputs, operators = self._get_inputs_and_operators(inputs_operators)  # extract the data
        if states is None:  # initialize the state vectors
            states = self.rnn.get_initial_state(inputs)
            states = [tf.cast(state, tf.float32) for state in states]

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

    def __init__(self, state_space,
                 checkpoint_B,
                 B=5, K=256, T=np.inf,
                 train_iterations=10,
                 reg_param=0.001,
                 controller_cells=48,
                 embedding_dim=30,
                 input_B=None,
                 pnas_mode=False,
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
            pnas_mode: if True, do not build a regressor to estimate time. Use only LSTM controller,
                like original PNAS.
            restore_controller: flag whether to restore a pre-trained RNN controller
                upon construction.
        '''
        self._logger = log_service.get_logger(__name__)
        self._amllibrary_logger = log_service.get_logger('aMLLibrary')

        self.state_space = state_space  # type: StateSpace
        self.state_size = self.state_space.size

        self.global_epoch = 0

        self.B = B
        self.K = K
        self.T = T
        self.embedding_dim = embedding_dim

        self.train_iterations = train_iterations
        self.controller_cells = controller_cells
        self.reg_strength = reg_param
        self.input_B = input_B
        self.pnas_mode = pnas_mode
        self.restore_controller = restore_controller

        self.build_regressor_config = True

        # restore controller
        # TODO: surely not working by beginning, it used csv files that don't exists!
        if self.restore_controller:
            self.b_ = checkpoint_B
            self._logger.info("Loading controller history!")

            next_children = []

            # read next_children from .csv file
            with open(log_service.build_path('csv', 'next_children.csv'), newline='') as f:
                reader = csv.reader(f, delimiter=',')
                for row in reader:
                    encoded_row = []
                    for i in range(len(row)):
                        if i % 2 == 0:
                            encoded_row.append(int(row[i]))
                        else:
                            encoded_row.append(row[i])
                    next_children.append(encoded_row)

            for i in range(1, self.b_):
                # read children from .csv file
                with open(log_service.build_path('csv', f'children_{i}.csv'), newline='') as f:
                    reader = csv.reader(f, delimiter=',')
                    j = 0
                    for row in reader:
                        encoded_row = []
                        for el in range(len(row)):
                            if el % 2 == 0:
                                encoded_row.append(int(row[el]))
                            else:
                                encoded_row.append(row[el])
                        np_encoded_row = np.array(encoded_row, dtype=np.object)
                        if j == 0:
                            children_i = [np_encoded_row]
                        else:
                            children_i = np.concatenate((children_i, [np_encoded_row]), axis=0)
                        j = j + 1

                # read old rewards from .csv file
                with open(log_service.build_path('csv', f'rewards_{i}.csv'), newline='') as f:
                    reader = csv.reader(f, delimiter=',')
                    j = 0
                    for row in reader:
                        if j == 0:
                            rewards_i = [float(row[0])]
                        else:
                            rewards_i.append(float(row[0]))
                        j = j + 1
                    rewards_i = np.array(rewards_i, dtype=np.float32)

                if i == 1:
                    children = [children_i]
                    rewards = [rewards_i]
                else:
                    children.append(children_i)
                    rewards.append(rewards_i)

            self.state_space.update_children(next_children)
            self.children_history = children

            self.score_history = rewards

        else:
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

        if self.restore_controller and self.input_B is not None:
            input_B = self.input_B
        else:
            input_B = self.state_space.inputs_embedding_max

        self.global_step = tf.compat.v1.train.get_or_create_global_step()
        #learning_rate = tf.compat.v1.train.exponential_decay(0.001, self.global_step, 500, 0.98, staircase=True)

        self.controller = Controller(self.controller_cells,
                                        self.embedding_dim,
                                        input_B,
                                        self.state_space.operator_embedding_max)

        # PNAS paper specifies different learning rates, one for b=1 and another for other b values
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.002)
        self.optimizer_b1 = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)

        self.saver = tf.train.Checkpoint(controller=self.controller,
                                         optimizer=self.optimizer,
                                         optimizer_b1=self.optimizer_b1,
                                         global_step=self.global_step)

        if self.restore_controller:
            path = tf.train.latest_checkpoint(log_service.build_path('weights'))

            if path is not None and tf.train.checkpoint_exists(path):
                self._logger.info("Loading controller checkpoint!")
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
        self._logger.info("Rewards : %s", rewards)

        if self.children_history is None:
            self.children_history = [children]
            self.score_history = [rewards]
            batchsize = rewards.shape[0]
        else:
            self.children_history.append(children)
            self.score_history.append(rewards)
            batchsize = sum([data.shape[0] for data in self.score_history])

        train_size = batchsize * self.train_iterations
        self._logger.info("Controller: Number of training steps required for this stage : %d", train_size)

        # logs
        self.logdir = log_service.build_path('controller')
        summary_writer = tf.summary.create_file_writer(self.logdir)
        summary_writer.set_as_default()

        for current_epoch in range(1, self.train_iterations + 1):
            self._logger.info("Controller: Begin training epoch %d", current_epoch)

            self.global_epoch = self.global_epoch

            for current_b in range(1, self.b_ + 1):
                children = self.children_history[current_b - 1]
                scores = self.score_history[current_b - 1]
                ids = np.array(list(range(len(scores))))
                np.random.shuffle(ids)

                self._logger.info("Controller: Begin training - B = %d", current_b)
                epoch_losses = np.array([])

                pbar = tqdm(iterable=enumerate(zip(children[ids], scores[ids])),
                        unit='child',
                        desc=f'Training LSTM (B={current_b}, epoch={current_epoch}): ',
                        total=len(children[ids]))

                for _, (child, score) in pbar:
                    child = child.tolist()
                    state_list = self.state_space.entity_encode_child(child)
                    state_list = np.concatenate(state_list, axis=-1).astype('int32')

                    state_list = tf.convert_to_tensor(state_list)

                    with tf.GradientTape() as tape:
                        rnn_scores, states = self.controller(state_list, states=None)
                        acc_scores = score.reshape((1, 1))

                        loss = self.loss(acc_scores, rnn_scores)

                    grads = tape.gradient(loss, self.controller.trainable_variables)
                    grad_vars = zip(grads, self.controller.trainable_variables)

                    optimizer = self.optimizer_b1 if current_b == 1 else self.optimizer
                    optimizer.apply_gradients(grad_vars, self.global_step)

                    epoch_losses = np.append(epoch_losses, loss.numpy())

                avg_loss = np.mean(epoch_losses)

                self._logger.info("Controller: Finished training epoch %d / %d of B = %d / %d, loss: %0.6f",
                                  current_epoch, self.train_iterations, current_b, self.b_, avg_loss)

            # add accuracy to Tensorboard
            with summary_writer.as_default():
                tf.summary.scalar("average_accuracy", rewards.mean(), description="controller", step=self.global_epoch)
                tf.summary.scalar("average_loss", avg_loss, description="controller", step=self.global_epoch)

        with open(log_service.build_path('csv', 'rewards.csv'), mode='a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(map(lambda x: [x], rewards))

        # save weights
        self.saver.save(log_service.build_path('weights', 'controller.ckpt'))

        return avg_loss

    def setup_regressor(self, techniques=['NNLS']):
        '''
        Generate time regressor configuration and build the regressor.

        Returns:
            (Regressor): time regressor (aMLLibrary)
        '''
        # NNLS, SVR, XGBoost, LRRidge

        # create the regressor configuration file for aMLLibrary
        # done only at first call of this function
        if self.build_regressor_config:
            config = ConfigParser()
            config.read(os.path.join('configs', 'regressors.ini'))

            for section in config.sections():
                if section == 'General':
                    continue

                # delete config section not relevant to selected techniques
                if section not in techniques:
                    del config[section]

            # value in .ini must be a single string of format ['technique1', 'technique2', ...]
            # note: '' are important for correct execution (see map)
            techniques_iter = map(lambda s: f"'{s}'", techniques)
            techniques_str = f"[{', '.join(techniques_iter)}]"
            config['General']['techniques'] = techniques_str
            config['DataPreparation'] = {'input_path': log_service.build_path('csv', 'training_time.csv')}

            with open(log_service.build_path('ini', 'aMLLibrary_regressors.ini'), 'w') as f:
                config.write(f)
            self.build_regressor_config = False

        # a-MLLibrary, redirect output to POPNAS logger (it uses stderr for output, see custom logger)
        redir_logger = StreamToLogger(self._amllibrary_logger)
        with redirect_stdout(redir_logger):
            with redirect_stderr(redir_logger):
                sequence_data_processor = sequence_data_processing.SequenceDataProcessing(
                    log_service.build_path('ini', 'aMLLibrary_regressors.ini'),
                    output=log_service.build_path(f'output_regressor_B{self.b_}'))

                best_regressor = sequence_data_processor.process()

        return best_regressor

    def estimate_time(self, regressor, child_encoding, headers, reindex_function):
        '''
        Use regressor to estimate the time for training the model.

        Args:
            regressor (Regressor): time regressor
            child_encoding (list[str]): model encoding
            headers ([type]): [description]
            reindex_function ([type]): [description]

        Returns:
            (float): estimated time predicted
        '''
  
        # TODO: the f** is happening here?
        encoded_child = self.state_space.entity_encode_child(child_encoding)
        concatenated_child = np.concatenate(encoded_child, axis=None).astype('int32')
        reindexed_child = []
        for i, action_index in enumerate(concatenated_child):
            if i % 2 == 0:
                # TODO: investigate this (probably avoids 0 for -2 input, as 0 is also used for null)
                reindexed_child.append(action_index + 1)
            else:
                reindexed_child.append(reindex_function(action_index))
                
        regressor_features = np.concatenate(reindexed_child, axis=None)

        # add missing blocks num feature (see training_time.csv, all columns except time are needed)
        regressor_features = np.append(np.array([self.b_]), [x for x in regressor_features])
        headers = headers[1:]   # remove time, because it's the regressor output

        # complete features with missing blocks (0 is for null)
        for _ in range(self.b_, self.B):
            regressor_features = np.append(regressor_features, np.array([0, 0, 0, 0]))

        df_row = pandas.DataFrame([regressor_features], columns=headers)
        # array of single element (time prediction)
        predicted_time = regressor.predict(df_row)[0]

        return predicted_time

    def estimate_accuracy(self, child_encoding):
        '''
        Use RNN controller to estimate the model accuracy.

        Args:
            child_encoding (list[str]): model encoding

        Returns:
            (float): estimated accuracy predicted
        '''

        state_list = self.state_space.entity_encode_child(child_encoding)
        state_list = np.concatenate(state_list, axis=-1).astype('int32')
        state_list = tf.convert_to_tensor(state_list)

        score, _ = self.controller(state_list, states=None)
        score = score[0, 0].numpy()

        return score

    def __write_predictions_on_csv(self, model_estimates):
        '''
        Write predictions on csv for further data analysis.

        Args:
            model_estimates (list[ModelEstimate]): [description]
        '''
        with open(log_service.build_path('csv', f'predictions_B{self.b_}.csv'), mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time', 'val accuracy', 'cell structure'])
            writer.writerows(map(lambda model_est: model_est.to_csv_array(), model_estimates))

    def update_step(self, headers, reindex_function):
        '''
        Updates the children from the intermediate products for the next generation
        of larger number of blocks in each cell
        '''

        # plot controller architecture
        # plot_model(self.controller, to_file='%s/controller_plot.png' % self.logdir, show_shapes=True, show_layer_names=True)

        # TODO: pandas is used only to add 0s and remove headers? But this is already done in code...
        csv_path = log_service.build_path('csv', 'training_time.csv')
        df = pandas.read_csv(csv_path)

        df.to_csv(csv_path, na_rep=0, index=False)

        regressor_NNLS = self.setup_regressor(techniques=['NNLS'])

        if self.b_ + 1 <= self.B:
            self.b_ += 1
            model_estimations = []  # type: list[ModelEstimate]

            # closure that returns a function that returns the model generator for current generation step
            generate_models = self.state_space.prepare_intermediate_children(self.b_)

            # TODO: leave eqv models in estimation and prune them when extrapolating pareto front, so that it prunes only the 
            # necessary ones and takes lot less time (instead of O(N^2) it becomes O(len(pareto)^2)). Now done in that way,
            # if you can make it more performant prune them before the evaluations again.
            next_models = list(generate_models())

            # TODO: previous method with generator
            # pbar = tqdm(iterable=enumerate(generate_models()),
            #             unit='model', desc='Estimating models: ',
            #             total=self.state_space.get_current_step_total_models(self.b_))

            pbar = tqdm(iterable=next_models,
                        unit='model', desc='Estimating models: ',
                        total=len(next_models))

            # iterate through all the intermediate children (intermediate_child is an array of repeated [input,action,input,action] blocks)
            for intermediate_child in pbar:
                # Regressor (aMLLibrary, estimates time)
                estimated_time = None if self.pnas_mode \
                    else self.estimate_time(regressor_NNLS, intermediate_child, headers, reindex_function)

                # LSTM controller (RNN, estimates the accuracy)
                score = self.estimate_accuracy(intermediate_child)

                pbar.set_postfix({ 'score': score }, refresh=False)

                # always preserve the child and its score in pnas mode, otherwise check that time estimation is < T (time threshold)
                if self.pnas_mode or estimated_time <= self.T:
                    model_estimations.append(ModelEstimate(intermediate_child, score, estimated_time))

            # sort the children according to their score
            model_estimations = sorted(model_estimations, key=lambda x: x.score, reverse=True)
            self.__write_predictions_on_csv(model_estimations)

            self._logger.info('Model evaluation completed')

            # start process by putting first model into pareto front (best score, ordered array),
            # then comparing the rest only by time because of ordering trick.
            # Pareto front can be built only if using regressor (needs time estimation, not possible in pnas mode)
            if not self.pnas_mode:
                self._logger.info('Building pareto front...')
                pareto_front = [model_estimations[0]]

                # for eqv check purposes
                existing_model_reprs = []
                pruned_count = 0

                for model_est in model_estimations[1:]:
                    # less time than last pareto element
                    if model_est.time <= pareto_front[-1].time:
                        # check that model is not equivalent to another one present already in the pareto front
                        cell_repr = CellEncoding(model_est.model_encoding)
                        if not self.state_space.check_model_equivalence(cell_repr, existing_model_reprs):
                            pareto_front.append(model_est)
                            existing_model_reprs.append(cell_repr)
                        else:
                            pruned_count += 1

                self._logger.info('Pruned %d equivalent models while building pareto front', pruned_count) 

                with open(log_service.build_path('csv', f'pareto_front_B{self.b_}.csv'), mode='w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(ModelEstimate.get_csv_headers())
                    writer.writerows(map(lambda model_est: model_est.to_csv_array(), pareto_front))

                self._logger.info('Pareto front built successfully')    
            else:
                # just a rename to integrate with existent code below, it's not a pareto front in this case!
                pareto_front = model_estimations

            # account for case where there are fewer children than K
            children_count = len(pareto_front) if self.K is None else min(self.K, len(pareto_front))

            if not self.pnas_mode:
                # take only the K highest scoring children for next iteration
                children = list(map(lambda child: child.model_encoding, pareto_front[:children_count]))
            else:
                # remove equivalent models, not done already if running in pnas mode
                models = list(map(lambda model_est: model_est.model_encoding, pareto_front))
                children, pruned_count = self.state_space.prune_equivalent_cell_models(models, children_count)
                self._logger.info('Pruned %d equivalent models while selecting CNN children', pruned_count)

            with open(log_service.build_path('csv', 'children.csv'), mode='a+', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(children)

            # save these children for next round
            self.state_space.update_children(children)
        else:
            self._logger.info("No more updates necessary as max B has been reached!")


class ModelEstimate:
    '''
    Helper class, basically a struct with a function to convert into array for csv saving
    '''
    def __init__(self, model_encoding, score, time):
        self.model_encoding = model_encoding
        self.score = score
        self.time = time

    def to_csv_array(self):
        cell_structure = f"[{';'.join(map(lambda el: str(el), self.model_encoding))}]"
        return [self.time, self.score, cell_structure]

    @staticmethod
    def get_csv_headers():
        return ['time', 'val accuracy', 'cell structure']