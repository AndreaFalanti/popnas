import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from encoder import SearchSpace
from utils.func_utils import to_list_of_tuples


def __prepare_rnn_inputs_2i(search_space: SearchSpace, cell_spec: list):
    '''
    Splits a cell specification (list of [in, op]) into separate inputs
    and operators tensors to be used in LSTM.

    # Args:
        cell_spec: interleaved [input; operator] pairs, not encoded.

    # Returns:
        (tuple): contains list of inputs and list of operators.
    '''
    # use categorical encoding for both input and operators, since LSTM works with categorical
    cell_encoding = search_space.encode_cell_spec(cell_spec)

    inputs = cell_encoding[0::2]  # even place data
    operators = cell_encoding[1::2]  # odd place data

    # add sequence dimension (final shape is (B, 2)),
    # to process blocks one at a time by the LSTM (2 inputs, 2 operators)
    inputs = [[in1, in2] for in1, in2 in to_list_of_tuples(inputs, 2)]
    operators = [[op1, op2] for op1, op2 in to_list_of_tuples(operators, 2)]

    return [inputs, operators]


def __prepare_rnn_inputs(search_space: SearchSpace, cell_spec: list):
    '''
    Encode a cell specification to be used in LSTM.

    # Args:
        cell_spec: interleaved [input; operator] pairs, not encoded.

    # Returns:
        (tuple): contains list of inputs and list of operators.
    '''
    # use categorical encoding for both input and operators, since LSTM works with categorical
    return search_space.encode_cell_spec(cell_spec, flatten=False)


def __generate_equivalent_cells_and_rewards(search_space: SearchSpace, cell_specs: 'list[list]', rewards: 'list[float]' = None):
    # generate the equivalent cell specifications.
    # this provides a data augmentation mechanism that can help the LSTM to learn better.
    eqv_cell_specs, eqv_rewards = [], []

    # build dataset for training (y labels are present)
    for cell_spec, reward in zip(cell_specs, rewards):
        eqv_cells, _ = search_space.generate_eqv_cells(cell_spec)

        # add padding with None blocks, if cells have not the maximum capacity of blocks
        eqv_cells = [cell_spec + [(None, None, None, None)] * (search_space.B - len(cell_spec)) for cell_spec in eqv_cells]

        # add {len(eqv_cells)} repeated elements into the reward list
        eqv_rewards.extend([reward] * len(eqv_cells))
        eqv_cell_specs.extend(eqv_cells)

    # set original variables to data augmented ones
    return eqv_cell_specs, eqv_rewards


def __preprocess_cell_specs_and_rewards(search_space: SearchSpace, cell_specs: 'list[list]', rewards: 'list[float]', use_data_augmentation):
    # data augmentation is used only in training (rewards are given), if the respective flag is set. For prediction it's never used.
    if use_data_augmentation and rewards is not None:
        cell_specs, rewards = __generate_equivalent_cells_and_rewards(search_space, cell_specs, rewards)
    else:
        # add padding with None blocks, if cells have not the maximum capacity of blocks
        cell_specs = [cell_spec + [(None, None, None, None)] * (search_space.B - len(cell_spec)) for cell_spec in cell_specs]

    # change shape of the rewards to a 2-dim tensor, where the second dim is 1.
    if rewards is not None:
        rewards = np.array(rewards, dtype=np.float32)
        rewards = np.expand_dims(rewards, -1)

    return cell_specs, rewards


def build_temporal_series_dataset_2i(search_space: SearchSpace, cell_specs: 'list[list]', rewards: 'list[float]' = None, batch_size: int = 4,
                                     validation_split: bool = True, use_data_augmentation: bool = True, shuffle: bool = True):
    '''
    Build a dataset to be used in the RNN controller, 2 separate series for inputs and operators.

    Args:
        search_space:
        cell_specs (list): List of lists of inputs and operators, specification of cells in value form (no encoding).
        rewards (list[float], optional): List of rewards (y labels). Defaults to None, provide it for building
            a dataset for training purposes.
        validation_split: split the dataset in train and validation. If false, return a single dataset.
        use_data_augmentation:

    Returns:
        tf.data.Dataset: [description]
    '''
    def build_dataset(x_cell_specs, y_rewards):
        rnn_inputs = list(map(lambda cell: __prepare_rnn_inputs_2i(search_space, cell), x_cell_specs))
        # fit function actually wants two distinct lists, instead of a list of tuples. This does the trick.
        rnn_in = [inputs for inputs, _ in rnn_inputs]
        rnn_ops = [ops for _, ops in rnn_inputs]

        ds = tf.data.Dataset.from_tensor_slices((rnn_in, rnn_ops))
        if y_rewards is not None:
            ds_label = tf.data.Dataset.from_tensor_slices(y_rewards)
            ds = tf.data.Dataset.zip((ds, ds_label))
        else:
            # add fake y, otherwise the input will be separated instead of using a pair of tensors for unknown reasons
            fake_y = np.ones(shape=(len(x_cell_specs), 1))
            ds_label = tf.data.Dataset.from_tensor_slices(fake_y)
            ds = tf.data.Dataset.zip((ds, ds_label))

        if shuffle:
            ds = ds.shuffle(10000)

        return ds.batch(batch_size)

    cells_train, rewards_train = __preprocess_cell_specs_and_rewards(search_space, cell_specs, rewards, use_data_augmentation)
    if validation_split:
        cells_train, cells_val, rewards_train, rewards_val = train_test_split(cells_train, rewards_train, test_size=0.1, shuffle=True)
        return build_dataset(cells_train, rewards_train), build_dataset(cells_val, rewards_val)
    else:
        return build_dataset(cells_train, rewards_train), None


def build_temporal_series_dataset(search_space: SearchSpace, cell_specs: 'list[list]', rewards: 'list[float]' = None, batch_size: int = 4,
                                  validation_split: bool = True, use_data_augmentation: bool = True, shuffle: bool = True):
    '''
    Build a dataset to be used in the RNN controller, using whole blocks as a temporal serie.

    Args:
        search_space:
        cell_specs (list): List of lists of inputs and operators, specification of cells in value form (no encoding).
        rewards (list[float], optional): List of rewards (y labels). Defaults to None, provide it for building
            a dataset for training purposes.
        validation_split: split the dataset in train and validation. If false, return a single dataset.
        use_data_augmentation:

    Returns:
        tf.data.Dataset: [description]
    '''
    def build_dataset(x_cell_specs, y_rewards):
        cell_series = list(map(lambda cell: __prepare_rnn_inputs(search_space, cell), x_cell_specs))

        ds = tf.data.Dataset.from_tensor_slices(cell_series)
        if y_rewards is not None:
            ds_label = tf.data.Dataset.from_tensor_slices(y_rewards)
            ds = tf.data.Dataset.zip((ds, ds_label))

        if shuffle:
            ds = ds.shuffle(10000)

        return ds.batch(batch_size)

    cells_train, rewards_train = __preprocess_cell_specs_and_rewards(search_space, cell_specs, rewards, use_data_augmentation)
    if validation_split:
        cells_train, cells_val, rewards_train, rewards_val = train_test_split(cells_train, rewards_train, test_size=0.1, shuffle=True)
        return build_dataset(cells_train, rewards_train), build_dataset(cells_val, rewards_val)
    else:
        return build_dataset(cells_train, rewards_train), None
