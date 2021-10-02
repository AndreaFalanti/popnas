import argparse
import logging
import os
import shutil
import sys

import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import layers, regularizers, callbacks, optimizers, losses, metrics, Model
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import log_service
from utils.func_utils import to_list_of_tuples, parse_cell_structures, compute_spearman_rank_correlation_coefficient, compute_mape
from ..encoder import StateSpace

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # disable Tensorflow info messages


class ControllerManagerTesting:
    '''
    Utility class to easily test different configurations of the controller on existing data of a previous run.
    It is based on the actual controller manager, without the regressor part, therefore there is a good amount of duplication
    in some parts, but it's better to keep them separate to perform tests more easily and keeping the actual implementation decoupled
    from this script.
    '''

    def __init__(self, state_space: StateSpace, B: int, logger: logging.Logger,
                 train_iterations=10, reg_param=0.001, controller_cells=48, embedding_dim=30,
                 pnas_mode=False, use_previous_data=True):
        '''
        Manages the Controller network training and prediction process.

        # Args:
            state_space: completely defined search space.
            B: depth of progression.
            train_iterations: number of training epochs for the RNN per depth level.
            reg_param: strength of the L2 regularization loss.
            controller_cells: number of cells in the Controller LSTM.
            embedding_dim: embedding dimension for inputs and operators.
            pnas_mode: if True, do not build a regressor to estimate time. Use only LSTM controller,
                like original PNAS.
        '''
        self._logger = logger
        self.log_path = log_service.build_path('controller')

        self.state_space = state_space
        self.B = B

        self.global_epoch = 0

        self.embedding_dim = embedding_dim

        self.train_iterations = train_iterations
        self.controller_cells = controller_cells
        self.reg_strength = reg_param
        self.pnas_mode = pnas_mode
        self.use_previous_data = use_previous_data

        self.build_regressor_config = True

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

        ds = tf.data.Dataset.from_tensor_slices((rnn_in, rnn_ops))
        if rewards is not None:
            ds_label = tf.data.Dataset.from_tensor_slices(rewards)
            ds = tf.data.Dataset.zip((ds, ds_label))
        else:
            # TODO: add fake y, otherwise the input will be separated instead of using a pair of tensors...
            ds_label = tf.data.Dataset.from_tensor_slices([[1]])
            ds = tf.data.Dataset.zip((ds, ds_label))

        # add batch size (MUST be done here, if specified in .fit function it doesn't work!)
        # TODO: also shuffle data, it can be good for better train when reusing old data (should be verified with actual testing, but i suppose so)
        ds = ds.shuffle(10000).batch(1)

        # for element in ds:
        #     print(element)

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

        return Model(inputs=(inputs, ops), outputs=score)

    def define_callbacks(self, tb_logdir):
        '''
        Define callbacks used in model training.

        Returns:
            (tf.keras.Callback[]): Keras callbacks
        '''
        model_callbacks = []

        # By default shows losses and metrics for both training and validation
        # tb_callback = callbacks.TensorBoard(log_dir=tb_logdir, profile_batch=0, histogram_freq=0, update_freq='epoch')
        # model_callbacks.append(tb_callback)

        return model_callbacks

    def build_policy_network(self):
        '''
        Construct the RNN controller network with the provided settings.

        Also constructs saver and restorer to the RNN controller if required.
        '''

        # learning_rate = tf.compat.v1.train.exponential_decay(0.001, self.global_step, 500, 0.98, staircase=True)

        # TODO: L1 regularizer is cited in PNAS paper, but where to apply it?
        reg = regularizers.l1(self.reg_strength) if self.reg_strength > 0 else None
        self.controller = self.build_controller_model(reg)

        # PNAS paper specifies different learning rates, one for b=1 and another for other b values
        self.optimizer = optimizers.Adam(learning_rate=0.002)
        self.optimizer_b1 = optimizers.Adam(learning_rate=0.01)

    def train_step(self, b: int, cells, rewards):
        '''
        Perform a single train step on the Controller RNN
        '''
        # create the dataset using also previous data, if flag is set.
        # a list of values is stored for both cells and their rewards.
        if self.use_previous_data:
            self.children_history.extend(cells)
            self.score_history.extend(rewards)

            rnn_dataset = self.__build_rnn_dataset(self.children_history, self.score_history)
        # use only current data
        else:
            rnn_dataset = self.__build_rnn_dataset(cells, rewards)

        train_size = len(rnn_dataset) * self.train_iterations
        self._logger.info("Controller: Number of training steps required for this stage : %d", train_size)

        loss = losses.MeanSquaredError()
        train_metrics = [metrics.MeanAbsolutePercentageError()]
        optimizer = self.optimizer_b1 if b == 1 else self.optimizer
        model_callbacks = self.define_callbacks(self.log_path)

        # TODO: recompiling will reset optimizer values, don't know if optimizer for b > 1 should be reset or not.
        self.controller.compile(optimizer=optimizer, loss=loss, metrics=train_metrics)

        # train only on last trained CNN batch.
        # Controller starts from the weights trained on previous CNNs, so retraining on them would cause overfitting on previous samples.
        hist = self.controller.fit(
            x=rnn_dataset,
            epochs=self.train_iterations,
            callbacks=model_callbacks
        )

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


def setup_folders(log_path):
    controller_test_path = os.path.join(log_path, 'controllers_test')
    try:
        os.makedirs(controller_test_path)
    except OSError:
        shutil.rmtree(controller_test_path)
        os.makedirs(controller_test_path)

    return controller_test_path


def create_logger(name, log_path):
    logger = logging.getLogger(name)

    # Create handlers
    file_handler = logging.FileHandler(os.path.join(log_path, 'debug.log'))
    file_handler.setFormatter(logging.Formatter("%(asctime)s - [%(name)s:%(levelname)s] %(message)s"))
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])

    return logger


def plot_squared_scatter_chart(x, y, name, log_path, plot_reference=True, legend_labels=None):
    fig, ax = plt.subplots()

    # list of lists with same dimensions are required, or also flat lists with same dimensions
    assert len(x) == len(y)

    # list of lists case
    if any(isinstance(el, list) for el in x):
        assert len(x) == len(legend_labels)

        colors = cm.rainbow(np.linspace(0, 1, len(x)))
        for xs, ys, color, lab in zip(x, y, colors, legend_labels):
            plt.scatter(xs, ys, color=color, label=lab)
    else:
        plt.scatter(x, y)

    plt.xlabel('Real accuracy')
    plt.ylabel('Predicted accuracy')
    plt.title(f'Accuracy predictions overview ({name})')
    plt.gca().set_aspect('equal', adjustable='box')

    plt.legend(fontsize='x-small')

    # add reference line (bisector line x = y)
    if plot_reference:
        ax_lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]

        ax.plot(ax_lims, ax_lims, '--k', alpha=0.75)

    log_path = os.path.join(log_path, name)
    plt.savefig(log_path, bbox_inches='tight')


def read_run_training_data(training_data_df: pd.DataFrame, b: int):
    b_df = training_data_df[training_data_df['# blocks'] == b]

    cells = parse_cell_structures(b_df['cell structure'])

    # fix cell structure having inputs as str type instead of int
    adjusted_cells = []
    for cell in cells:
        adjusted_cells.append([(int(in1), op1, int(in2), op2) for in1, op1, in2, op2 in cell])

    # just return two lists: one with the accuracy, one with the cell structures
    return b_df['best val accuracy'].tolist(), adjusted_cells


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-p', metavar='FOLDER', type=str, help="log folder", required=True)
    args = parser.parse_args()

    log_path = setup_folders(args.p)
    log_service.set_log_path(log_path)
    logger = log_service.get_logger('controller_testing')

    csv_path = os.path.join(args.p, 'csv')
    training_data_path = os.path.join(csv_path, 'training_results.csv')
    training_data_df = pd.read_csv(training_data_path)
    training_data_df = training_data_df.drop(columns=['flops', 'total params', 'training time(seconds)'])

    operators = ['identity', '3x3 dconv', '5x5 dconv', '7x7 dconv', '1x7-7x1 conv', '3x3 conv', '3x3 maxpool', '3x3 avgpool']
    b_max = 5
    state_space = StateSpace(B=b_max, operators=operators, input_lookback_depth=-2)

    controller_configs = [
        ControllerManagerTesting(state_space, b_max, logger, train_iterations=15, reg_param=0),
        # ControllerManagerTesting(state_space, b_max, logger, train_iterations=20),
        ControllerManagerTesting(state_space, b_max, logger, train_iterations=15, reg_param=1e-5),
        ControllerManagerTesting(state_space, b_max, logger, train_iterations=15, reg_param=1e-4)
    ]

    # create an empty list of dicts
    predictions = []
    for _ in controller_configs:
        predictions.append({'x': [], 'y': [], 'MAPE': [], 'spearman': []})

    scatter_plot_legend_labels = []
    for b in range(1, b_max):
        logger.info('--------------------------- B = %d -------------------------', b)
        # label of predictions
        scatter_plot_legend_labels.append(f'B{b + 1}')

        rewards, cells = read_run_training_data(training_data_df, b)
        logger.info('Extracted training data for step B=%d', b)

        for index, (controller_manager, preds_dict) in enumerate(zip(controller_configs, predictions)):
            logger.info('Starting training for config %d...', index)
            controller_manager.train_step(b, cells, rewards)

            pred_rewards, pred_cells = read_run_training_data(training_data_df, b + 1)
            logger.info('Starting predictions for config %d...', index)
            estimation_rewards = list(map(controller_manager.estimate_accuracy, pred_cells))

            preds_dict['x'].append(pred_rewards)
            preds_dict['y'].append(estimation_rewards)
            preds_dict['MAPE'].append(compute_mape(pred_rewards, estimation_rewards))

            # spearman can be retrieved with pandas, but dataframe must be built first
            spearman_df = pd.DataFrame.from_dict({'true': pred_rewards, 'est': estimation_rewards})
            preds_dict['spearman'].append(compute_spearman_rank_correlation_coefficient(spearman_df, 'true', 'est'))
            logger.info('Predictions and metrics produced successfully for config %d...', index)

        logger.info('--------------------------------------------------------------')

    logger.info('Building plots for each controller configuration')
    for index, (controller_manager, preds_dict) in enumerate(zip(controller_configs, predictions)):
        # add MAPE and spearman to legend
        legend_labels = list(map(lambda label, mape, spearman: label + f' (MAPE: {mape:.3f}%, ρ: {spearman:.3f}',
                                 scatter_plot_legend_labels, preds_dict['MAPE'], preds_dict['spearman']))

        plot_squared_scatter_chart(preds_dict['x'], preds_dict['y'], f'config_{index}',
                                   log_path, legend_labels=legend_labels)

    logger.info('Script completed successfully')


if __name__ == '__main__':
    main()
