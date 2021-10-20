import os
from logging import Logger
from typing import Union, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers, callbacks, optimizers, losses, metrics, Model
from tensorflow.keras.utils import plot_model
import pandas as pd

from encoder import StateSpace
from predictor import Predictor
from utils.func_utils import to_list_of_tuples, parse_cell_structures
from utils.rstr import rstr


class LSTMPredictor(Predictor):
    def __init__(self, state_space: StateSpace, y_col: str, y_domain: 'tuple[float, float]', logger: Logger, log_folder: str, name: str = None,
                 embedding_dim: int = 10, lstm_cells: int = 48, weight_reg: float = 1e-5, lr: float = 0.002, epochs: int = 15,
                 use_previous_data: bool = True):
        # generate a relevant name if not set
        if name is None:
            name = f'LSTM_ed({embedding_dim})_c({lstm_cells})_wr({weight_reg})_lr({lr})_e({epochs})_prev({use_previous_data})'
        super().__init__(logger, log_folder, name)

        self.state_space = state_space
        self.y_col = y_col
        self.embedding_dim = embedding_dim
        self.lstm_cells = lstm_cells
        self.epochs = epochs
        self.use_previous_data = use_previous_data

        # used to accumulate samples in a common dataset (a list for each B), if use_previous_data is True
        self.children_history = []
        self.score_history = []

        # choose the correct activation for last layer, based on y domain
        self.output_activation = None
        lower_bound, upper_bound = y_domain
        assert lower_bound < upper_bound, 'Invalid domain'

        # from tests relu is bad for (0, inf) domains. If you want to do more tests, check infinite bounds with math.isinf()
        if lower_bound == 0 and upper_bound == 1:
            self.output_activation = 'sigmoid'
        elif lower_bound == -1 and upper_bound == 1:
            self.output_activation = 'tanh'
        else:
            self.output_activation = 'linear'

        self._logger.info('Using %s as final activation, based on y domain provided', self.output_activation)

        self.loss = losses.MeanSquaredError()
        self.train_metrics = [metrics.MeanAbsolutePercentageError()]
        self.optimizer = optimizers.Adam(learning_rate=lr)
        self.weight_reg = regularizers.l2(weight_reg) if weight_reg > 0 else None
        self.callbacks = [
            callbacks.TensorBoard(log_dir=self.log_folder, profile_batch=0, histogram_freq=0, update_freq='epoch'),
            callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1, mode='min', restore_best_weights=True)
        ]

        self.model = self.__build_model()
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.train_metrics)

        plot_model(self.model, to_file=os.path.join(self.log_folder, 'model.png'), show_shapes=True, show_layer_names=True)

    def __build_model(self):
        # two inputs: one tensor for cell inputs, one for cell operators (both of 1-dim)
        # since the length varies, None is given as dimension
        inputs = layers.Input(shape=(self.state_space.B, 2))
        ops = layers.Input(shape=(self.state_space.B, 2))

        # input dim is the max integer value present in the embedding + 1.
        inputs_embed = layers.Embedding(input_dim=self.state_space.inputs_embedding_max, output_dim=self.embedding_dim,
                                        embeddings_regularizer=self.weight_reg, mask_zero=True)(inputs)
        ops_embed = layers.Embedding(input_dim=self.state_space.operator_embedding_max, output_dim=self.embedding_dim,
                                     embeddings_regularizer=self.weight_reg, mask_zero=True)(ops)

        embed = layers.Concatenate()([inputs_embed, ops_embed])
        # pass from (None, self.B, 2, 2*embedding_dim) to (None, self.B, 4*embedding_dim),
        # indicating [batch_size, serie_length, features(whole block embedding)]
        embed = layers.Reshape((self.state_space.B, 4 * self.embedding_dim))(embed)

        # attention = layers.Attention()([ops_embed, inputs_embed])
        # embed = layers.Reshape((self.B, 2 * self.embedding_dim))(attention)

        # many-to-one, so must have return_sequences = False (it is by default)
        lstm = layers.Bidirectional(layers.LSTM(self.lstm_cells, kernel_regularizer=self.weight_reg, recurrent_regularizer=self.weight_reg))(embed)
        score = layers.Dense(1, activation=self.output_activation, kernel_regularizer=self.weight_reg)(lstm)

        return Model(inputs=(inputs, ops), outputs=score)

    def __prepare_rnn_inputs(self, cell_spec):
        '''
        Splits a cell specification (list of [in, op]) into separate inputs
        and operators tensors to be used in LSTM.

        # Args:
            cell_spec: interleaved [input; operator] pairs, not encoded.

        # Returns:
            (tuple): contains list of inputs and list of operators.
        '''
        # use categorical encoding for both input and operators, since LSTM works with categorical
        cell_encoding = self.state_space.encode_cell_spec(cell_spec)

        inputs = cell_encoding[0::2]  # even place data
        operators = cell_encoding[1::2]  # odd place data

        # add sequence dimension (final shape is (B, 2)),
        # to process blocks one at a time by the LSTM (2 inputs, 2 operators)
        inputs = [[in1, in2] for in1, in2 in to_list_of_tuples(inputs, 2)]
        operators = [[op1, op2] for op1, op2 in to_list_of_tuples(operators, 2)]

        # right padding to reach B elements
        for i in range(len(inputs), self.state_space.B):
            inputs.append([0, 0])
            operators.append([0, 0])

        return [inputs, operators]

    def __build_rnn_dataset(self, cell_specs: 'list[list]', rewards: 'list[float]' = None, use_data_augmentation: bool = True):
        '''
        Build a dataset to be used in the RNN controller.

        Args:
            cell_specs (list): List of lists of inputs and operators, specification of cells in value form (no encoding).
            rewards (list[float], optional): List of rewards (y labels). Defaults to None, provide it for building
                a dataset for training purposes.

        Returns:
            tf.data.Dataset: [description]
        '''
        # data augmentation is used only in training (rewards are given), if the respective flag is set.
        # if data augment is performed, the cell_specs and rewards parameters are replaced with their augmented counterpart.
        if use_data_augmentation and rewards is not None:
            # generate the equivalent cell specifications.
            # this provides a data augmentation mechanism that can help the LSTM to learn better.
            eqv_cell_specs, eqv_rewards = [], []

            # build dataset for training (y labels are present)
            for cell_spec, reward in zip(cell_specs, rewards):
                eqv_cells, _ = self.state_space.generate_eqv_cells(cell_spec)

                # add {len(eqv_cells)} repeated elements into the reward list
                eqv_rewards.extend([reward] * len(eqv_cells))
                eqv_cell_specs.extend(eqv_cells)

            # set original variables to data augmented ones
            cell_specs = eqv_cell_specs
            rewards = eqv_rewards

        # change shape of the rewards to a 2-dim tensor, where the second dim is 1.
        if rewards is not None:
            rewards = np.array(rewards, dtype=np.float32)
            rewards = np.expand_dims(rewards, -1)

        rnn_inputs = list(map(self.__prepare_rnn_inputs, cell_specs))
        # fit function actually wants two distinct lists, instead of a list of tuples. This does the trick.
        rnn_in = [inputs for inputs, _ in rnn_inputs]
        rnn_ops = [ops for _, ops in rnn_inputs]

        ds = tf.data.Dataset.from_tensor_slices((rnn_in, rnn_ops))
        if rewards is not None:
            ds_label = tf.data.Dataset.from_tensor_slices(rewards)
            ds = tf.data.Dataset.zip((ds, ds_label))
        else:
            # TODO: add fake y, otherwise the input will be separated instead of using a pair of tensors... Better ideas?
            ds_label = tf.data.Dataset.from_tensor_slices([[1]])
            ds = tf.data.Dataset.zip((ds, ds_label))

        # add batch size (MUST be done here, if specified in .fit function it doesn't work!)
        # TODO: also shuffle data, it can be good for better train when reusing old data (should be verified with actual testing, but i suppose so)
        ds = ds.shuffle(10000).batch(1)

        # DEBUG
        # for element in ds:
        #     print(element)

        return ds

    def __get_max_b(self, df: pd.DataFrame):
        return df['# blocks'].max()

    def __extrapolate_samples_for_b(self, training_data_df: pd.DataFrame, b: int):
        b_df = training_data_df[training_data_df['# blocks'] == b]

        cells = parse_cell_structures(b_df['cell structure'])

        # fix cell structure having inputs as str type instead of int
        adjusted_cells = []
        for cell in cells:
            adjusted_cells.append([(int(in1), op1, int(in2), op2) for in1, op1, in2, op2 in cell])

        # just return two lists: one with the target, one with the cell structures
        return b_df[self.y_col].tolist(), adjusted_cells

    def train(self, dataset: Union[str, 'list[Tuple]'], use_data_augmentation=True):
        # TODO
        if not isinstance(dataset, list):
            raise TypeError('LSTM supports only samples, conversion from file is a TODO...')

        cells, rewards = zip(*dataset)

        # create the dataset using also previous data, if flag is set.
        # a list of values is stored for both cells and their rewards.
        if self.use_previous_data:
            self.children_history.extend(cells)
            self.score_history.extend(rewards)

            rnn_dataset = self.__build_rnn_dataset(self.children_history, self.score_history, use_data_augmentation)
        # use only current data
        else:
            rnn_dataset = self.__build_rnn_dataset(cells, rewards, use_data_augmentation)

        train_size = len(rnn_dataset) * self.epochs
        self._logger.info("Controller: Number of training steps required for this stage : %d", train_size)

        # Controller starts from the weights learned from previous training sessions, since it's not re-instantiated.
        hist = self.model.fit(x=rnn_dataset,
                              epochs=self.epochs,
                              callbacks=self.callbacks)
        self._logger.info("losses: %s", rstr(hist.history['loss']))

    def predict(self, sample: list) -> float:
        pred_dataset = self.__build_rnn_dataset([sample])
        return self.model.predict(x=pred_dataset)[0, 0]

    def prepare_prediction_test_data(self, file_path: str) -> 'tuple[list[Union[str, list[tuple]]], list[list], list[list[float]]]':
        dataset_df = pd.read_csv(file_path)
        max_b = self.__get_max_b(dataset_df)
        drop_columns = [col for col in dataset_df.columns.values.tolist() if col not in [self.y_col, 'cell structure', '# blocks']]
        dataset_df = dataset_df.drop(columns=drop_columns)

        datasets = []
        prediction_samples = []
        real_values = []

        for b in range(1, max_b):
            targets, cells = self.__extrapolate_samples_for_b(dataset_df, b)
            datasets.append(list(zip(cells, targets)))

            true_values, cells_to_predict = self.__extrapolate_samples_for_b(dataset_df, b + 1)

            real_values.append(true_values)
            prediction_samples.append(cells_to_predict)

        return datasets, prediction_samples, real_values
