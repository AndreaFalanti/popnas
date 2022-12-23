import os
from abc import ABC
from logging import Logger
from typing import Union, Iterable, Optional, Sequence

import keras_tuner as kt
import numpy as np
import sklearn
import spektral
import tensorflow as tf
from einops import rearrange
from sklearn.model_selection import train_test_split
from spektral import layers as g_layers
from spektral.data import Loader, Graph, PackedBatchLoader
from tensorflow.keras import losses, optimizers, metrics, callbacks
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import plot_model

from encoder import SearchSpace
from predictors import KerasPredictor
from utils.func_utils import create_empty_folder, to_one_hot, list_flatten, chunks
from utils.rstr import rstr


def build_adjacency_matrix(edges: 'Iterable[tuple[int, int]]', num_nodes: int):
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float)
    for i, j in edges:
        adj_matrix[i][j] = 1

    return adj_matrix


def cell_spec_to_spektral_graph(search_space: SearchSpace, cell_spec: list, y_label: Optional[float]):
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

    return Graph(x=node_features, a=adj, y=y_label)


class CellGraphDataset(spektral.data.Dataset):
    def __init__(self, search_space: SearchSpace, cell_specs: 'list[list]', rewards: 'list[float]' = None, transforms=None, **kwargs):
        self.search_space = search_space
        self.cell_specs = cell_specs
        self.rewards = rewards if rewards is not None else [None] * len(cell_specs)

        super().__init__(transforms, **kwargs)

    def read(self):
        return [cell_spec_to_spektral_graph(self.search_space, cell_spec, reward) for cell_spec, reward in zip(self.cell_specs, self.rewards)]

    # data is always local and passed directly to __init__, no need to implement this
    def download(self):
        pass


class ExtractLastNodeFeatures(Layer):
    def __init__(self, node_dim: int, name='last_node_feat', **kwargs):
        '''
        Take only last node features from features produced by a Spektral graph layer, reshaping it to squeeze dimensions.
        Encapsulating the operation in a Keras layer allows to plot it during plot_model().
        '''
        super().__init__(name=name, **kwargs)
        self.sort_pool = g_layers.SortPool(1)
        self.node_dim = node_dim

    def call(self, inputs, training=None, mask=None):
        last_node_feat = self.sort_pool(inputs)
        return rearrange(last_node_feat, 'b n f -> b (n f)', n=1, f=self.node_dim)

    def get_config(self):
        config = super().get_config()
        config.update({
            'node_dim': self.node_dim,
        })
        return config


class SpektralPredictor(KerasPredictor, ABC):
    def __init__(self, search_space: SearchSpace, y_col: str, y_domain: 'tuple[float, float]', train_strategy: tf.distribute.Strategy, logger: Logger,
                 log_folder: str, name: str = None, override_logs: bool = True, save_weights: bool = False,
                 hp_config: dict = None, hp_tuning: bool = False):
        super().__init__(search_space, y_col, y_domain, train_strategy, logger, log_folder, name, override_logs, save_weights, hp_config, hp_tuning)

        self.search_space = search_space
        # Spektral transformations to apply on GraphDataset to satisfy eventual model requirements or improvements
        # (e.g. GCNFilter for GCN conv based models)
        # They can be used on datasets with .apply(tx) or directly to a Graph object (apply simply iterates over dataset graphs).
        self.data_transforms = self._setup_data_transforms()

        # node features are (in order): operators as one-hot categorical, lookbacks, block_add and (cell_concat + pointwise conv)
        # features are a one-hot vector, which should work well for GCN.
        num_operators = len(self.search_space.operator_values)
        max_lookback_abs = abs(self.search_space.input_lookback_depth)
        self.num_node_features = num_operators + max_lookback_abs + 2

    def _setup_data_transforms(self) -> list:
        '''
        Prepare a list of Spektral transformations that must be applied to the samples for correct model usage. If not overridden,
        no transformation is applied.
        '''
        return []

    def _prepare_graph_for_predict(self, cell_spec: list):
        g = cell_spec_to_spektral_graph(self.search_space, cell_spec, None)
        for tx in self.data_transforms:
            g = tx(g)

        # expand dims to fake the batch size first dimension (set to 1).
        return np.expand_dims(g['x'], axis=0), np.expand_dims(g['a'], axis=0)

    def _build_tf_dataset(self, cell_specs: 'Sequence[list]', rewards: 'Sequence[float]' = None, batch_size: int = 8,
                          use_data_augmentation: bool = True, validation_split: bool = True,
                          shuffle: bool = True) -> 'tuple[Loader, Optional[Loader]]':
        # NOTE: instead of TF Datasets, this function returns spektral loaders, which are generators. Still, they should work fine in Keras functions.

        # prune empty cell, since it's not possible to feed an empty graph.
        if cell_specs[0] == []:
            cell_specs = cell_specs[1:]
            rewards = None if rewards is None else rewards[1:]

        # TODO: WORKAROUND -> predict case, avoid loader and use numpy directly, to avoid bug with keras model predict
        if len(cell_specs) == 1:
            g = self._prepare_graph_for_predict(cell_specs[0])
            return g, None

        train_gds, val_gds = None, None
        if validation_split:
            cells_train, cells_val, rewards_train, rewards_val = train_test_split(cell_specs, rewards, test_size=0.1, shuffle=shuffle)
            train_gds = CellGraphDataset(self.search_space, cells_train, rewards_train)
            val_gds = CellGraphDataset(self.search_space, cells_val, rewards_val)
        else:
            cell_specs, rewards = sklearn.utils.shuffle(cell_specs, rewards)
            train_gds = CellGraphDataset(self.search_space, cell_specs, rewards)

        for tx in self.data_transforms:
            train_gds.apply(tx)
            if val_gds is not None:
                val_gds.apply(tx)

        return PackedBatchLoader(train_gds, batch_size=batch_size, shuffle=False, node_level=False).load(), \
               None if val_gds is None else PackedBatchLoader(val_gds, batch_size=batch_size, shuffle=False, node_level=False).load()

    def _get_compilation_parameters(self, lr: float) -> 'tuple[losses.Loss, optimizers.Optimizer, list[metrics.Metric]]':
        loss = losses.MeanSquaredError()
        train_metrics = [metrics.MeanAbsolutePercentageError()]
        optimizer = optimizers.Adam(learning_rate=lr)

        return loss, optimizer, train_metrics

    def _get_callbacks(self, log_folder: str) -> 'list[callbacks.Callback]':
        return [
            callbacks.TensorBoard(log_dir=log_folder, profile_batch=0, histogram_freq=0, update_freq='epoch'),
            callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1, mode='min', restore_best_weights=True)
        ]

    def train(self, dataset: Union[str, 'list[tuple]'], use_data_augmentation=True):
        # get samples for file path string
        if isinstance(dataset, str):
            dataset = self._get_training_data_from_file(dataset)

        cells, rewards = zip(*dataset)
        actual_b = len(cells[-1])
        # erase ensemble in case single training is used. If called from train_ensemble, the ensemble is local to the function and written
        # only at the end
        self.model_ensemble = None

        batch_size = 8
        train_ds, val_ds = self._build_tf_dataset(cells, rewards, batch_size,
                                                  use_data_augmentation=use_data_augmentation, validation_split=self.hp_tuning)

        if self._model_log_folder is None:
            b_max = len(cells[-1])
            self._model_log_folder = os.path.join(self.log_folder, f'B{b_max}')
            create_empty_folder(self._model_log_folder)

        train_callbacks = self._get_callbacks(self._model_log_folder)
        self._logger.info('Starting Keras predictor training')

        if self.hp_tuning:
            tuner_callbacks = [callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1, mode='min', restore_best_weights=True)]
            tuner = kt.Hyperband(self._compile_model, objective='val_loss', hyperparameters=self._get_hp_search_space(),
                                 max_epochs=30,
                                 directory=os.path.join(self.log_folder, 'keras-tuner'), project_name=f'B{actual_b}')
            tuner.search(x=train_ds,
                         epochs=self.hp_config['epochs'],
                         steps_per_epoch=train_ds.steps_per_epoch,
                         validation_data=val_ds,
                         validation_steps=val_ds.steps_per_epoch,
                         callbacks=tuner_callbacks)
            best_hp = tuner.get_best_hyperparameters()[0].values
            self._logger.info('Best hyperparameters found: %s', rstr(best_hp))

            # train the best model with all samples
            # TODO: is there a way to merge them without a complete rebuild?
            whole_ds, _ = self._build_tf_dataset(cells, rewards, batch_size,
                                                 use_data_augmentation=use_data_augmentation, validation_split=False)

            self.model = self._compile_model(best_hp)
            self.model.fit(x=whole_ds,
                           epochs=self.hp_config['epochs'],
                           steps_per_epoch=train_ds.steps_per_epoch,
                           callbacks=train_callbacks)   # type: callbacks.History
        else:
            self.model = self._compile_model(self.hp_config)
            self.model.fit(x=train_ds,
                           epochs=self.hp_config['epochs'],
                           steps_per_epoch=train_ds.steps_per_epoch,
                           callbacks=train_callbacks)

        plot_model(self.model, to_file=os.path.join(self.log_folder, 'model.pdf'), show_shapes=True, show_layer_names=True)
        self._logger.info('Keras predictor trained successfully')

        if self.save_weights:
            self.model.save_weights(os.path.join(self._model_log_folder, 'weights'))
