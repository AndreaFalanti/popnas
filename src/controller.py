import csv
import os
from configparser import ConfigParser
from contextlib import redirect_stderr, redirect_stdout
from typing import Union

import psutil
import catboost
import numpy as np
import pandas
import tensorflow as tf
from tensorflow.keras import layers, optimizers, losses, regularizers, metrics, callbacks, Model
from tensorflow.keras.utils import plot_model
from tqdm import tqdm

import cell_pruning
import log_service
from aMLLibrary import sequence_data_processing
from aMLLibrary.regressor import Regressor
from encoder import StateSpace
from utils.stream_to_logger import StreamToLogger
from utils.func_utils import to_list_of_tuples, strip_unused_amllibrary_config_sections


class ControllerManager:
    '''
    Utility class to manage the RNN Controller.

    Tasked with maintaining the state of the training schedule,
    keep track of the children models generated from cross-products,
    cull non-optimal children model configurations and resume
    training.
    '''

    def __init__(self, state_space: StateSpace, checkpoint_B,
                 B=5, K=256, T=np.inf,
                 train_iterations=10, reg_param=1e-4, controller_cells=48, embedding_dim=30, lr1=0.01, lr2=0.002,
                 pnas_mode=False, use_previous_data=True,
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
            pnas_mode: if True, do not build a regressor to estimate time. Use only LSTM controller,
                like original PNAS.
            restore_controller: flag whether to restore a pre-trained RNN controller
                upon construction.
        '''
        self._logger = log_service.get_logger(__name__)
        self._amllibrary_logger = log_service.get_logger('aMLLibrary')
        self._catboost_logger = log_service.get_logger('CatBoost')
        self.log_path = log_service.build_path('controller')

        self.state_space = state_space  # type: StateSpace

        self.global_epoch = 0

        self.B = B
        self.K = K
        self.T = T

        self.embedding_dim = embedding_dim
        self.train_iterations = train_iterations
        self.controller_cells = controller_cells
        self.reg_strength = reg_param
        self.lr1 = lr1
        self.lr2 = lr2

        self.pnas_mode = pnas_mode
        self.restore_controller = restore_controller
        self.use_previous_data = use_previous_data

        self.build_regressor_config = True

        # restore controller
        # TODO: surely not working by beginning, it used csv files that don't exist!
        if self.restore_controller:
            # region fix_restore_mess
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
            # endregion
        else:
            self.b_ = 1
            self.children_history = []
            self.score_history = []

        self.build_policy_network()

    def __prepare_rnn_inputs(self, cell_spec):
        '''
        Splits a cell specification (list of [in, op]) into separate inputs
        and operators tensors to be used in LSTM.

        # Args:
            cell_spec: interleaved [input; operator] pairs, not encoded.

        # Returns:
            (tuple): contains list of inputs and list of operators.
        '''
        cell_encoding = self.state_space.encode_cell_spec(cell_spec)

        inputs = cell_encoding[0::2]  # even place data
        operators = cell_encoding[1::2]  # odd place data

        # add sequence dimension (final shape is (B, 2)),
        # to process blocks one at a time by the LSTM (2 inputs, 2 operators)
        inputs = [[in1, in2] for in1, in2 in to_list_of_tuples(inputs, 2)]
        operators = [[op1, op2] for op1, op2 in to_list_of_tuples(operators, 2)]

        # right padding to reach B elements
        for i in range(len(inputs), self.B):
            inputs.append([0, 0])
            operators.append([0, 0])

        return [inputs, operators]

    def __build_rnn_dataset(self, cell_specs: 'list[list]', rewards: 'list[float]' = None):
        '''
        Build a dataset to be used in the RNN controller.

        Args:
            cell_specs (list): List of lists of inputs and operators, specification of cells in value form (no encoding). 
            rewards (list[float], optional): List of rewards (y labels). Defaults to None, provide it for building
                a dataset for training purposes.

        Returns:
            tf.data.Dataset: [description]
        '''
        # generate also equivalent cell specifications. This provides a data augmentation mechanism that
        # can help the LSTM to learn better
        eqv_cell_specs, eqv_rewards = [], []

        # build dataset for predictions (no y labels), simply rename the list without doing data augmentation
        if rewards is None:
            eqv_cell_specs = cell_specs
        # build dataset for training (y labels are present)
        else:
            for cell_spec, reward in zip(cell_specs, rewards):
                eqv_cells, _ = self.state_space.generate_eqv_cells(cell_spec)

                # add {len(eqv_cells)} repeated elements into the reward list
                eqv_rewards.extend([reward] * len(eqv_cells))
                eqv_cell_specs.extend(eqv_cells)

            rewards = np.array(eqv_rewards, dtype=np.float32)
            rewards = np.expand_dims(rewards, -1)

        rnn_inputs = list(map(lambda child: self.__prepare_rnn_inputs(child), eqv_cell_specs))
        # fit function actually wants two distinct lists, instead of a list of tuples. This does the trick.
        rnn_in = [inputs for inputs, _ in rnn_inputs]
        rnn_ops = [ops for _, ops in rnn_inputs]

        ds = tf.data.Dataset.from_tensor_slices(({"input_1": rnn_in, "input_2": rnn_ops})) if rewards is None \
            else tf.data.Dataset.from_tensor_slices(({"input_1": rnn_in, "input_2": rnn_ops}, rewards))

        # add batch size (MUST be done here, if specified in .fit function it doesn't work!)
        # TODO: also shuffle data, it can be good for better train when reusing old data (should be verified with actual testing, but i suppose so)
        ds = ds.shuffle(10000).batch(1)

        return ds

    def build_controller_model(self, weight_reg):
        # two inputs: one tensor for cell inputs, one for cell operators (both of 1-dim)
        # since the length varies, None is given as dimension
        inputs = layers.Input(shape=(self.B, 2))
        ops = layers.Input(shape=(self.B, 2))

        # input dim is the max integer value present in the embedding + 1.
        inputs_embed = layers.Embedding(input_dim=self.state_space.inputs_embedding_max, output_dim=self.embedding_dim)(inputs)
        ops_embed = layers.Embedding(input_dim=self.state_space.operator_embedding_max, output_dim=self.embedding_dim)(ops)

        embed = layers.Concatenate()([inputs_embed, ops_embed])
        # pass from (None, self.B, 2, 2*embedding_dim) to (None, self.B, 4*embedding_dim),
        # indicating [batch_size, serie_length, features(whole block embedding)]
        embed = layers.Reshape((self.B, 4 * self.embedding_dim))(embed)

        # many-to-one, so must have return_sequences = False (it is by default)
        lstm = layers.LSTM(self.controller_cells, kernel_regularizer=weight_reg, recurrent_regularizer=weight_reg)(embed)
        score = layers.Dense(1, activation='sigmoid', kernel_regularizer=weight_reg)(lstm)

        return Model(inputs=[inputs, ops], outputs=score)

    def define_callbacks(self, tb_logdir):
        '''
        Define callbacks used in model training.

        Returns:
            (tf.keras.Callback[]): Keras callbacks
        '''
        model_callbacks = []

        # By default shows losses and metrics for both training and validation
        tb_callback = callbacks.TensorBoard(log_dir=tb_logdir, profile_batch=0, histogram_freq=0, update_freq='epoch')
        model_callbacks.append(tb_callback)

        plateau_callback = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.4, patience=4, verbose=1)
        model_callbacks.append(plateau_callback)

        return model_callbacks

    def build_policy_network(self):
        '''
        Construct the RNN controller network with the provided settings.

        Also constructs saver and restorer to the RNN controller if required.
        '''

        # learning_rate = tf.compat.v1.train.exponential_decay(0.001, self.global_step, 500, 0.98, staircase=True)

        # TODO: L1 regularizer is cited in PNAS paper, but where to apply it?
        reg = regularizers.l1(self.reg_strength)
        self.controller = self.build_controller_model(reg)

        with open(os.path.join(self.log_path, 'summary.txt'), 'w') as f:
            self.controller.summary(line_length=150, print_fn=lambda x: f.write(x + '\n'))

        plot_model(self.controller, to_file=os.path.join(self.log_path, 'model.png'), show_shapes=True, show_layer_names=True)

        # PNAS paper specifies different learning rates, one for b=1 and another for other b values
        self.optimizer = optimizers.Adam(learning_rate=self.lr2)
        self.optimizer_b1 = optimizers.Adam(learning_rate=self.lr1)

        self.saver = tf.train.Checkpoint(controller=self.controller,
                                         optimizer=self.optimizer,
                                         optimizer_b1=self.optimizer_b1)

        if self.restore_controller:
            path = tf.train.latest_checkpoint(os.path.join(self.log_path, 'weights'))

            if path is not None and tf.train.latest_checkpoint(path):
                self._logger.info("Loading controller checkpoint!")
                self.saver.restore(path)

    def train_step(self, rewards):
        '''
        Perform a single train step on the Controller RNN

        # Returns:
            final training loss
        '''
        # create the dataset using also previous data, if flag is set.
        # a list of values is stored for both cells and their rewards.
        if self.use_previous_data:
            self.children_history.extend(self.state_space.children)
            self.score_history.extend(rewards)

            rnn_dataset = self.__build_rnn_dataset(self.children_history, self.score_history)
        # use only current data
        else:
            rnn_dataset = self.__build_rnn_dataset(self.state_space.children, rewards)

        train_size = len(rnn_dataset) * self.train_iterations
        self._logger.info("Controller: Number of training steps required for this stage : %d", train_size)

        loss = losses.MeanSquaredError()
        train_metrics = [metrics.MeanAbsolutePercentageError()]
        optimizer = self.optimizer_b1 if self.b_ == 1 else self.optimizer
        model_callbacks = self.define_callbacks(self.log_path)

        # TODO: recompiling will reset optimizer values, don't know if optimizer for b > 1 should be reset or not.
        self.controller.compile(optimizer=optimizer, loss=loss, metrics=train_metrics)

        # train only on last trained CNN batch.
        # Controller starts from the weights trained on previous CNNs, so retraining on them would cause overfitting on previous samples.
        hist = self.controller.fit(x=rnn_dataset,
                                   epochs=self.train_iterations,
                                   callbacks=model_callbacks)

        with open(log_service.build_path('csv', 'rewards.csv'), mode='a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(map(lambda x: [x], rewards))

        # save weights
        self.saver.save(log_service.build_path('controller', 'weights', 'controller.ckpt'))

        final_loss = hist.history['loss'][-1]
        return final_loss

    def __write_regressor_config_file(self, techniques):
        config = ConfigParser()
        # to keep casing in keys while reading / writing
        config.optionxform = str

        config.read(os.path.join('configs', 'regressors_hyperopt.ini'))

        strip_unused_amllibrary_config_sections(config, techniques)

        # value in .ini must be a single string of format ['technique1', 'technique2', ...]
        # note: '' are important for correct execution (see map)
        techniques_iter = map(lambda s: f"'{s}'", techniques)
        techniques_str = f"[{', '.join(techniques_iter)}]"
        config['General']['techniques'] = techniques_str
        config['DataPreparation']['input_path'] = log_service.build_path('csv', 'training_time.csv')

        with open(log_service.build_path('ini', 'aMLLibrary_regressors.ini'), 'w') as f:
            config.write(f)
        self.build_regressor_config = False

    def setup_regressor(self, techniques=['NNLS']):
        '''
        Generate time regressor configuration and build the regressor (aMLLibrary).
        Supported techniques: NNLS, SVR, XGBoost, LRRidge

        Returns:
            (Regressor): time regressor (aMLLibrary)
        '''

        # create the regressor configuration file for aMLLibrary
        # done only at first call of this function
        if self.build_regressor_config:
            self.__write_regressor_config_file(techniques)

        # a-MLLibrary, redirect output to POPNAS logger (it uses stderr for output, see custom logger)
        redir_logger = StreamToLogger(self._amllibrary_logger)
        usable_cpu_cores = len(psutil.Process().cpu_affinity())
        self._logger.info("Running regressors training on %d threads", usable_cpu_cores)
        with redirect_stdout(redir_logger):
            with redirect_stderr(redir_logger):
                sequence_data_processor = sequence_data_processing.SequenceDataProcessing(
                    log_service.build_path('ini', 'aMLLibrary_regressors.ini'),
                    output=log_service.build_path('regressors', f'B{self.b_}'),
                    j=usable_cpu_cores
                )

                best_regressor = sequence_data_processor.process()

        return best_regressor

    def train_catboost_regressor(self, input_csv_path: str, train_log_path: str):
        column_description_file = log_service.build_path('csv', 'column_desc_time.csv')
        # first feature is the one used as y (time)
        train_pool = catboost.Pool(input_csv_path, delimiter=',', has_header=True, column_description=column_description_file)

        # param_grid = {
        #     'learning_rate': [0.1, 0.15],
        #     'depth': [4, 5, 6, 7],
        #     'l2_leaf_reg': [1, 3, 6],
        #     'random_strength': [1.2, 1.6],
        #     'bagging_temperature': [0.4],
        #     'grow_policy': ['SymmetricTree', 'Lossguide']
        # }

        redir_logger = StreamToLogger(self._catboost_logger)
        with redirect_stdout(redir_logger):
            with redirect_stderr(redir_logger):
                # specify the training parameters
                regressor = catboost.CatBoostRegressor(custom_metric='MAPE', early_stopping_rounds=16, train_dir=train_log_path)
                # train the model
                # results_dict = regressor.grid_search(param_grid, train_pool, train_size=0.85)
                regressor.fit(train_pool)

        # self._logger.info('Best parameters: %s', str(results_dict['params']))
        return regressor

    def estimate_time(self, regressor: Union[Regressor, catboost.CatBoostRegressor], child_spec: list, headers: 'list[str]'):
        '''
        Use regressor to estimate the time for training the model.

        Args:
            regressor (Regressor): time regressor
            child_spec (list[str]): model encoding
            headers ([type]): [description]

        Returns:
            (float): estimated time predicted
        '''
        # regressor uses dynamic reindex for operations, instead of categorical
        encoded_child = self.state_space.encode_cell_spec(child_spec, op_enc_name='dynamic_reindex')

        # add missing blocks num feature (see training_time.csv, all columns except time are needed)
        regressor_features = np.append(np.array([self.b_]), encoded_child)
        headers = headers[1:-1]  # remove time, because it's the regressor output, and data_augmented field (last one)

        # complete features with missing blocks (0 is for null)
        for _ in range(self.b_, self.B):
            regressor_features = np.append(regressor_features, np.array([0, 0, 0, 0]))

        # TODO: could be made more "elegant", but right now is the best way to avoid more sophisticated changes
        if isinstance(regressor, catboost.CatBoostRegressor):
            cat_features_indexes = [indexes for i in range(self.B) for indexes in (4 * i + 1, 4 * i + 3)]

            # ndarray doesn't support multiple types, so are all floats, but categorical must be int or string
            regressor_features = regressor_features.tolist()
            for cat_index in cat_features_indexes:
                regressor_features[cat_index] = int(regressor_features[cat_index])

            features_pool = catboost.Pool([regressor_features], cat_features=cat_features_indexes, feature_names=headers)
            predicted_time = regressor.predict(features_pool)[0]
        else:
            df_row = pandas.DataFrame([regressor_features], columns=headers)
            # array of single element (time prediction)
            predicted_time = regressor.predict(df_row)[0]

        return predicted_time

    def estimate_accuracy(self, child_spec: 'list[tuple]'):
        '''
        Use RNN controller to estimate the model accuracy.

        Args:
            child_spec (list[tuple]): plain cell specification

        Returns:
            (float): estimated accuracy predicted
        '''
        # TODO: Dataset of single element, maybe not much efficient...
        pred_dataset = self.__build_rnn_dataset([child_spec])

        score = self.controller.predict(x=pred_dataset)
        # score is a numpy array of shape (1, 1) since model has a single output (return_sequences=False)
        # and dataset has a single item. Simply return the plain element.
        return score[0, 0]

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

    def update_step(self, headers):
        '''
        Updates the children from the intermediate products for the next generation
        of larger number of blocks in each cell
        '''

        csv_path = log_service.build_path('csv', 'training_time.csv')

        # TODO: choice between catboost and aMLLibrary. Choose best settings based on tuning runs.
        regressor = self.setup_regressor(techniques=['LRRidge']) if self.b_ == 1 \
            else self.train_catboost_regressor(csv_path, log_service.build_path('regressors', f'B{self.b_}'))

        if self.b_ + 1 <= self.B:
            self.b_ += 1
            model_estimations = []  # type: list[ModelEstimate]

            # closure that returns a function that returns the model generator for current generation step
            generate_models = self.state_space.prepare_intermediate_children(self.b_)

            # TODO: leave eqv models in estimation and prune them when extrapolating pareto front, so that it prunes only the
            #  necessary ones and takes lot less time (instead of O(N^2) it becomes O(len(pareto)^2)). Now done in that way,
            #  if you can make it more performant prune them before the evaluations again.
            next_models = list(generate_models())

            pbar = tqdm(iterable=next_models,
                        unit='model', desc='Estimating models: ',
                        total=len(next_models))

            # iterate through all the intermediate children (intermediate_child is a list of repeated (input,action,input,action) tuple blocks)
            for intermediate_child in pbar:
                # Regressor (aMLLibrary, estimates time)
                estimated_time = None if self.pnas_mode \
                    else self.estimate_time(regressor, intermediate_child, headers)

                # LSTM controller (RNN, estimates the accuracy)
                score = self.estimate_accuracy(intermediate_child)

                pbar.set_postfix({'score': score}, refresh=False)

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
                        cell_repr = cell_pruning.CellEncoding(model_est.model_encoding)
                        if not cell_pruning.check_model_equivalence(cell_repr, existing_model_reprs):
                            pareto_front.append(model_est)
                            existing_model_reprs.append(cell_repr)
                        else:
                            pruned_count += 1

                self._logger.info('Pruned %d equivalent models while building pareto front', pruned_count)

                with open(log_service.build_path('csv', f'pareto_front_B{self.b_}.csv'), mode='w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(ModelEstimate.get_csv_headers())
                    writer.writerows(map(lambda est: est.to_csv_array(), pareto_front))

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
                models = list(map(lambda est: est.model_encoding, pareto_front))
                children, pruned_count = cell_pruning.prune_equivalent_cell_models(models, children_count)
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
