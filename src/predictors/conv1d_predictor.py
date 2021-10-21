import os
from logging import Logger

import tensorflow as tf
from tensorflow.keras import layers, regularizers, callbacks, optimizers, losses, metrics, Model
from tensorflow.keras.utils import plot_model

from encoder import StateSpace
from nn_predictor import NNPredictor
from predictors.common.datasets_gen import build_temporal_serie_dataset_2i


class Conv1DPredictor(NNPredictor):
    def __init__(self, state_space: StateSpace, y_col: str, y_domain: 'tuple[float, float]', logger: Logger, log_folder: str, name: str = None,
                 epochs: int = 15, use_previous_data: bool = True, lr: float = 0.002, weight_reg: float = 1e-5,
                 filters: int = 12, kernel_size: int = 2):
        # generate a relevant name if not set
        if name is None:
            name = f'Conv1D_kernel({kernel_size})_f({filters})_wr({weight_reg})_lr({lr})_e({epochs})_prev({use_previous_data})'
        super().__init__(y_col, y_domain, logger, log_folder, name, epochs=epochs, use_previous_data=use_previous_data)

        self.state_space = state_space
        self.kernel_size = kernel_size
        self.filters = filters

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

        inputs_temp_conv = layers.Conv1D(self.filters, self.kernel_size, activation='relu', kernel_regularizer=self.weight_reg)(inputs)
        ops_temp_conv = layers.Conv1D(self.filters, self.kernel_size, activation='relu', kernel_regularizer=self.weight_reg)(ops)

        # indicating [batch_size, serie_length, features(whole block embedding)]
        block_serie = layers.Concatenate()([inputs_temp_conv, ops_temp_conv])

        block_temp_conv = layers.Conv1D(self.filters * 2, self.kernel_size, activation='relu', kernel_regularizer=self.weight_reg)(block_serie)

        flatten = layers.Flatten()(block_temp_conv)
        score = layers.Dense(1, activation=self.output_activation, kernel_regularizer=self.weight_reg)(flatten)

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
        return build_temporal_serie_dataset_2i(self.state_space, cell_specs, rewards, use_data_augmentation)