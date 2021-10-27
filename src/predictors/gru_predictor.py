import os
from logging import Logger

import tensorflow as tf
from tensorflow.keras import layers, regularizers, callbacks, optimizers, losses, metrics, Model
from tensorflow.keras.utils import plot_model

from encoder import StateSpace
from nn_predictor import NNPredictor
from predictors.common.datasets_gen import build_temporal_series_dataset_2i


class GRUPredictor(NNPredictor):
    def __init__(self, state_space: StateSpace, y_col: str, y_domain: 'tuple[float, float]', logger: Logger, log_folder: str, name: str = None,
                 embedding_dim: int = 10, rnn_cells: int = 48, weight_reg: float = 1e-5, lr: float = 0.002, epochs: int = 15,
                 use_previous_data: bool = True):
        # generate a relevant name if not set
        if name is None:
            name = f'GRU_ed({embedding_dim})_c({rnn_cells})_wr({weight_reg})_lr({lr})_e({epochs})_prev({use_previous_data})'
        super().__init__(y_col, y_domain, logger, log_folder, name, epochs=epochs, use_previous_data=use_previous_data)

        self.state_space = state_space
        self.embedding_dim = embedding_dim
        self.gru_cells = rnn_cells

        self.loss = losses.MeanSquaredError()
        self.train_metrics = [metrics.MeanAbsolutePercentageError()]
        self.optimizer = optimizers.Adam(learning_rate=lr)
        self.weight_reg = regularizers.l2(weight_reg) if weight_reg > 0 else None
        self.callbacks = [
            callbacks.TensorBoard(log_dir=self.log_folder, profile_batch=0, histogram_freq=0, update_freq='epoch'),
            callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1, mode='min', restore_best_weights=True)
        ]

        self.model = self._build_model()
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.train_metrics)

        plot_model(self.model, to_file=os.path.join(self.log_folder, 'model.png'), show_shapes=True, show_layer_names=True)

    def _build_model(self):
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
        lstm = layers.Bidirectional(layers.GRU(self.gru_cells, kernel_regularizer=self.weight_reg, recurrent_regularizer=self.weight_reg))(embed)
        score = layers.Dense(1, activation=self.output_activation, kernel_regularizer=self.weight_reg)(lstm)

        return Model(inputs=(inputs, ops), outputs=score)

    def _build_tf_dataset(self, cell_specs: 'list[list]', rewards: 'list[float]' = None, use_data_augmentation: bool = True):
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
        return build_temporal_series_dataset_2i(self.state_space, cell_specs, rewards, use_data_augmentation)
