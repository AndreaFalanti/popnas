from logging import Logger
from typing import Iterable

import numpy as np
import spektral
import spektral.layers as g_layers
import tensorflow as tf
from spektral.data import Graph, BatchLoader
from spektral.transforms import GCNFilter
from tensorflow.keras import activations, layers, Model

from encoder import SearchSpace
from keras_predictor import KerasPredictor
from utils.func_utils import alternative_dict_to_string, list_flatten, chunks, to_one_hot


def build_adjacency_matrix(edges: 'Iterable[tuple[int, int]]', num_nodes: int):
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float)

    for i, j in edges:
        adj_matrix[i][j] = 1

    return adj_matrix


def cell_spec_to_spektral_graph(search_space: SearchSpace, cell_spec: list, y_label: float):
    # address empty cell case (0 blocks), doesn't work
    # if len(cell_spec) == 0:
    #     return Graph(x=[], a=[], y=y_label)

    flat_cell = list_flatten(cell_spec)
    cell_inputs = flat_cell[::2]
    cell_ops = flat_cell[1::2]

    encoded_flat_cell = search_space.encode_cell_spec(cell_spec)
    encoded_cell_inputs = encoded_flat_cell[::2]
    encoded_cell_ops = [x - 1 for x in encoded_flat_cell[1::2]]  # change it to 0-indexed

    used_lookbacks = set(inp for inp in cell_inputs if inp < 0)
    used_blocks = set(inp for inp in cell_inputs if inp >= 0)
    num_blocks = len(cell_spec)
    unused_blocks = set(b for b in range(num_blocks) if b not in used_blocks)
    num_unused_blocks = len(unused_blocks)
    num_operators = len(search_space.operator_values)
    max_lookback_abs = abs(search_space.input_lookback_depth)

    # node features are (in order): operators as one-hot categorical, lookbacks, block_add and (cell_concat + pointwise conv)
    # features are a one-hot vector, which should work well for GCN.
    num_features = num_operators + max_lookback_abs + 2
    num_nodes = len(used_lookbacks) + num_blocks * 3 + (1 if num_unused_blocks > 1 else 0)
    num_edges = num_blocks * 4 + (num_unused_blocks if num_unused_blocks > 1 else 0)

    # nodes are 0-indexed, 0 is the further lookback input and so on.
    # After accommodating all lookback inputs, a triplet of nodes is created for each block.
    lookback_node_ids = []
    curr_lb_node = 0
    for i in range(max_lookback_abs):
        lookback_node_ids.append(curr_lb_node if curr_lb_node - max_lookback_abs in used_lookbacks else None)
        curr_lb_node += 1

    block_join_node_ids = [(i + 1) * 3 + len(used_lookbacks) - 1 for i in range(num_blocks)]
    input_node_ids = block_join_node_ids + lookback_node_ids
    cell_concat_node_id = num_nodes - 1

    lb_node_id_offset = len(used_lookbacks)
    op_node_ids = []
    for i in range(num_blocks):
        op_node_ids.extend([lb_node_id_offset + 3 * i, lb_node_id_offset + 3 * i + 1])

    # edges from inputs to op nodes
    edges_out = [input_node_ids[inp] for inp in cell_inputs]
    edges_in = op_node_ids.copy()

    # edges from op nodes to block join operator (the two block operators are connected to block join)
    edges_out.extend(op_node_ids)
    edges_in.extend(n for n_id in block_join_node_ids for n in [n_id] * 2)  # double for to duplicate values

    # add edges to cell concat, if necessary
    if num_unused_blocks > 1:
        edges_out.extend(block_join_node_ids[b] for b in unused_blocks)
        edges_in += [cell_concat_node_id] * num_unused_blocks

    # assign a number to each node, which will be converted layer to one-hot encoding.
    # for operator nodes, the categorical value is already available (transposed to 0-index)
    # op categorical values are followed by values for lookbacks, block join operator and cell concat.
    # node ids start with lookback nodes, process them in order. lb_value is negative.
    features_categorical = [num_operators + max_lookback_abs + lb_value for lb_value in sorted(used_lookbacks)]
    # add feature for block triplets
    block_features = list_flatten([(op1, op2, num_features - 2) for op1, op2 in chunks(encoded_cell_ops, 2)])
    features_categorical.extend(block_features)
    # add feature for cell concat, if present
    if num_unused_blocks > 1:
        features_categorical.append(num_features - 1)

    # map categorical values to one-hot
    node_features = [to_one_hot(cat_value, num_features) for cat_value in features_categorical]
    node_features = np.asarray(node_features, dtype=np.float)   # shape: [num_nodes, num_features]

    adj = build_adjacency_matrix(zip(edges_out, edges_in), num_nodes=num_nodes)  # shape: [num_nodes, num_nodes]

    g = Graph(x=node_features, a=adj, y=y_label)
    return g


class CellGraphDataset(spektral.data.Dataset):
    def __init__(self, search_space: SearchSpace, cell_specs: 'list[list]', rewards: 'list[float]' = None, transforms=None, **kwargs):
        self.search_space = search_space
        self.cell_specs = cell_specs
        self.rewards = rewards if rewards is not None else [None] * len(cell_specs)

        super().__init__(transforms, **kwargs)

    def read(self):
        return [cell_spec_to_spektral_graph(self.search_space, cell_spec, reward) for cell_spec, reward in zip(self.cell_specs, self.rewards)]

    def download(self):
        pass


class GNNModel(Model):
    def __init__(self, f1: int, f2: int, **kwargs):
        super().__init__(**kwargs)

        self.gconv1 = g_layers.GCNConv(f1, activation=activations.swish)
        self.gconv2 = g_layers.GCNConv(f2, activation=activations.swish)
        self.dense = layers.Dense(100, activation=activations.swish)
        self.out = layers.Dense(1, activation=activations.sigmoid)

    def call(self, inputs, training=None, mask=None):
        # destructure inputs into node_features and adjacency_matrix
        x, a = inputs

        x = self.gconv1([x, a])
        x = self.gconv2([x, a])
        # use only last node features
        x = self.dense(x)
        return self.out(x)

    # TODO
    def get_config(self):
        pass


class GCNPredictor(KerasPredictor):
    def __init__(self, search_space: SearchSpace, y_col: str, y_domain: 'tuple[float, float]', train_strategy: tf.distribute.Strategy,
                 logger: Logger, log_folder: str, name: str = None, override_logs: bool = True,
                 save_weights: bool = False, hp_config: dict = None, hp_tuning: bool = False):
        # generate a relevant name if not set
        if name is None:
            name = f'GCN_{"default" if hp_config is None else alternative_dict_to_string(hp_config)}_{"tune" if hp_tuning else "manual"}'

        super().__init__(y_col, y_domain, train_strategy, logger, log_folder, name, override_logs, save_weights, hp_config, hp_tuning)

        self.search_space = search_space
        # node features are (in order): operators as one-hot categorical, lookbacks, block_add and (cell_concat + pointwise conv)
        # features are a one-hot vector, which should work well for GCN.
        num_operators = len(self.search_space.operator_values)
        max_lookback_abs = abs(self.search_space.input_lookback_depth)
        self.num_features = num_operators + max_lookback_abs + 2

    # TODO
    def _get_default_hp_config(self):
        return dict(super()._get_default_hp_config(), **{
            'wr': 1e-5,
            'use_er': False,
            'er': 0,
            'cells': 48,
            'embedding_dim': 10,
            'rnn_type': 'lstm'
        })

    # TODO
    def _get_hp_search_space(self):
        hp = super()._get_hp_search_space()
        hp.Float('wr', 1e-7, 1e-4, sampling='log')
        hp.Boolean('use_er')
        hp.Float('er', 1e-7, 1e-4, sampling='log', parent_name='use_er', parent_values=[True])
        hp.Int('cells', 20, 100, sampling='linear')
        hp.Int('embedding_dim', 10, 100, sampling='linear')

        return hp

    def _build_model(self, config: dict):
        return GNNModel(f1=20, f2=30)

    def _build_tf_dataset(self, cell_specs: 'list[list]', rewards: 'list[float]' = None, batch_size: int = 8, use_data_augmentation: bool = True,
                          validation_split: bool = True, shuffle: bool = True):
        # NOTE: instead of TF Datasets, this function returns spektral loaders, which are generators. Still, they should work fine in Keras functions.

        # prune empty cell, since it's not possible to feed an empty graph.
        if cell_specs[0] == []:
            cell_specs = cell_specs[1:]
            rewards = None if rewards is None else rewards[1:]

        if validation_split:
            # TODO
            raise NotImplementedError()
        else:
            train_gds = CellGraphDataset(self.search_space, cell_specs, rewards)
            train_gds.apply(GCNFilter())
            # shuffle should not be applied here, it gives a warning
            return BatchLoader(train_gds, batch_size=batch_size, shuffle=False).load(), None
