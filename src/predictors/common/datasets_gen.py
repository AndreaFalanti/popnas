import numpy as np
import tensorflow as tf

from encoder import StateSpace
from utils.func_utils import to_list_of_tuples


def __prepare_rnn_inputs(state_space: StateSpace, cell_spec: list):
    '''
    Splits a cell specification (list of [in, op]) into separate inputs
    and operators tensors to be used in LSTM.

    # Args:
        cell_spec: interleaved [input; operator] pairs, not encoded.

    # Returns:
        (tuple): contains list of inputs and list of operators.
    '''
    # use categorical encoding for both input and operators, since LSTM works with categorical
    cell_encoding = state_space.encode_cell_spec(cell_spec)

    inputs = cell_encoding[0::2]  # even place data
    operators = cell_encoding[1::2]  # odd place data

    # add sequence dimension (final shape is (B, 2)),
    # to process blocks one at a time by the LSTM (2 inputs, 2 operators)
    inputs = [[in1, in2] for in1, in2 in to_list_of_tuples(inputs, 2)]
    operators = [[op1, op2] for op1, op2 in to_list_of_tuples(operators, 2)]

    # right padding to reach B elements
    for i in range(len(inputs), state_space.B):
        inputs.append([0, 0])
        operators.append([0, 0])

    return [inputs, operators]


def build_temporal_serie_dataset_2i(state_space: StateSpace, cell_specs: 'list[list]', rewards: 'list[float]' = None,
                                    use_data_augmentation: bool = True):
    '''
    Build a dataset to be used in the RNN controller, 2 separate series for inputs and operators.

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
            eqv_cells, _ = state_space.generate_eqv_cells(cell_spec)

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

    rnn_inputs = list(map(lambda cell: __prepare_rnn_inputs(state_space, cell), cell_specs))
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
