import os
from logging import Logger

os.environ['DGLBACKEND'] = 'tensorflow'

import dgl
from dgl.nn.tensorflow import GraphConv
import tensorflow as tf
from tensorflow.keras import activations, layers, Model

from encoder import SearchSpace
from keras_predictor import KerasPredictor
from utils.func_utils import alternative_dict_to_string, list_flatten, chunks, to_one_hot


class GNNModel(Model):
    def __init__(self, num_input_features: int, f1: int, f2: int, **kwargs):
        super().__init__(**kwargs)

        self.gconv1 = GraphConv(num_input_features, f1, activation=activations.swish)
        self.gconv2 = GraphConv(f1, f2, activation=activations.swish)
        self.dense = layers.Dense(100, activation=activations.swish)
        self.out = layers.Dense(1, activation=activations.sigmoid)

    def call(self, g, training=None, mask=None):
        x = self.gconv1(g, g.ndata['feat'])
        x = self.gconv2(g, x)
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
        return GNNModel(self.num_features, f1=20, f2=30)

    def _build_tf_dataset(self, cell_specs: 'list[list]', rewards: 'list[float]' = None, batch_size: int = 8, use_data_augmentation: bool = True,
                          validation_split: bool = True, shuffle: bool = True):
        # dataset = dgl.data.CoraGraphDataset()
        #
        # # A DGL dataset may contain multiple graphs.
        # # In the case of Cora, there is only one graph.
        # g = dataset[0]
        #
        # # g.ndata is a dictionary of nodes related data.
        # # Prepare the training, validation, and test datasets.
        # ds = tf.data.Dataset.zip((
        #     tf.data.Dataset.from_tensor_slices(g.ndata['feat'][g.ndata['train_mask']]),
        #     tf.data.Dataset.from_tensor_slices(g.ndata['label'][g.ndata['train_mask']])
        # )).batch(64)

        # skip empty cell
        graphs = [self.cell_spec_to_dgl_graph(cell_spec) for cell_spec in cell_specs[1:]]
        ds = tf.data.Dataset.zip((
            tf.data.Dataset.from_tensor_slices(graphs),
            tf.data.Dataset.from_tensor_slices(rewards)
        ))

        return ds

    def cell_spec_to_dgl_graph(self, cell_spec: list):
        flat_cell = list_flatten(cell_spec)
        cell_inputs = flat_cell[::2]
        cell_ops = flat_cell[1::2]

        encoded_flat_cell = self.search_space.encode_cell_spec(cell_spec)
        encoded_cell_inputs = encoded_flat_cell[::2]
        encoded_cell_ops = [x - 1 for x in encoded_flat_cell[1::2]]  # change it to 0-indexed

        used_lookbacks = set(inp for inp in cell_inputs if inp < 0)
        used_blocks = set(inp for inp in cell_inputs if inp >= 0)
        num_blocks = len(cell_spec)
        unused_blocks = set(b for b in range(num_blocks) if b not in used_blocks)
        num_unused_blocks = len(unused_blocks)
        num_operators = len(self.search_space.operator_values)
        max_lookback_abs = abs(self.search_space.input_lookback_depth)

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

        g = dgl.graph((edges_out, edges_in), num_nodes=num_nodes)
        g.ndata['feat'] = tf.constant(node_features)

        return g
