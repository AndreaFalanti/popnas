from logging import Logger

import keras.backend as K
import tensorflow as tf
from tensorflow.keras import activations, initializers, constraints, regularizers, layers
from tensorflow.keras.layers import Layer

from encoder import SearchSpace
from keras_predictor import KerasPredictor
from predictors.common.datasets_gen import build_temporal_series_dataset_2i
from utils.func_utils import alternative_dict_to_string


class GraphConvolution(Layer):
    """Basic graph convolution layer as in https://arxiv.org/abs/1609.02907"""
    def __init__(self, units, support=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(GraphConvolution, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.supports_masking = True

        self.support = support
        assert support >= 1

    def compute_output_shape(self, input_shapes):
        features_shape = input_shapes[0]
        output_shape = (features_shape[0], self.units)
        return output_shape  # (batch_size, output_dim)

    def build(self, input_shapes):
        features_shape = input_shapes[0]
        input_dim = features_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim * self.support,
                                             self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, mask=None):
        features = inputs[0]
        basis = inputs[1:]

        supports = list()
        for i in range(self.support):
            supports.append(K.dot(tf.sparse.to_dense(basis[i]), features))
        supports = K.concatenate(supports, axis=1)
        output = K.dot(supports, self.kernel)

        if self.bias is not None:
            output += self.bias
        return self.activation(output)

    def get_config(self):
        config = {'units': self.units,
                  'support': self.support,
                  'activation': activations.serialize(self.activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(
                      self.kernel_initializer),
                  'bias_initializer': initializers.serialize(
                      self.bias_initializer),
                  'kernel_regularizer': regularizers.serialize(
                      self.kernel_regularizer),
                  'bias_regularizer': regularizers.serialize(
                      self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(
                      self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(
                      self.kernel_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint)
        }

        base_config = super(GraphConvolution, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GCNPredictor(KerasPredictor):
    def __init__(self, search_space: SearchSpace, y_col: str, y_domain: 'tuple[float, float]', train_strategy: tf.distribute.Strategy,
                 logger: Logger, log_folder: str, name: str = None, override_logs: bool = True,
                 save_weights: bool = False, hp_config: dict = None, hp_tuning: bool = False):
        # generate a relevant name if not set
        if name is None:
            name = f'GCN_{"default" if hp_config is None else alternative_dict_to_string(hp_config)}_{"tune" if hp_tuning else "manual"}'

        super().__init__(y_col, y_domain, train_strategy, logger, log_folder, name, override_logs, save_weights, hp_config, hp_tuning)

        self.search_space = search_space
        self.supported_rnn_classes = {
            'lstm': layers.LSTM,
            'gru': layers.GRU
        }

    def _get_default_hp_config(self):
        return dict(super()._get_default_hp_config(), **{
            'wr': 1e-5,
            'use_er': False,
            'er': 0,
            'cells': 48,
            'embedding_dim': 10,
            'rnn_type': 'lstm'
        })

    def _get_hp_search_space(self):
        hp = super()._get_hp_search_space()
        hp.Float('wr', 1e-7, 1e-4, sampling='log')
        hp.Boolean('use_er')
        hp.Float('er', 1e-7, 1e-4, sampling='log', parent_name='use_er', parent_values=[True])
        hp.Int('cells', 20, 100, sampling='linear')
        hp.Int('embedding_dim', 10, 100, sampling='linear')
        hp.Choice('rnn_type', ['lstm', 'gru'])

        return hp

    def _build_model(self, config: dict):
        weight_reg = regularizers.l2(config['wr']) if config['wr'] > 0 else None
        embedding_reg = regularizers.l2(config['er']) if config['use_er'] else None
        rnn = self.supported_rnn_classes[config['rnn_type']]

        # two inputs: one tensor for cell inputs, one for cell operators
        inputs = layers.Input(shape=(self.search_space.B, 2))
        ops = layers.Input(shape=(self.search_space.B, 2))

        # input dim is the max integer value present in the embedding + 1.
        inputs_embed = layers.Embedding(input_dim=self.search_space.inputs_embedding_max, output_dim=config['embedding_dim'],
                                        embeddings_regularizer=embedding_reg, mask_zero=True)(inputs)
        ops_embed = layers.Embedding(input_dim=self.search_space.operator_embedding_max, output_dim=config['embedding_dim'],
                                     embeddings_regularizer=embedding_reg, mask_zero=True)(ops)

        embed = layers.Concatenate()([inputs_embed, ops_embed])
        # pass from 4D (None, B, 2, 2 * embedding_dim) to 3D (None, B, 4 * embedding_dim),
        # indicating [batch_size, serie_length, features(whole block embedding)]
        embed = layers.Reshape((self.search_space.B, 4 * config['embedding_dim']))(embed)

        # many-to-one, so must have return_sequences = False (it is by default)
        lstm = layers.Bidirectional(rnn(config['cells'], kernel_regularizer=weight_reg, recurrent_regularizer=weight_reg))(embed)
        score = layers.Dense(1, activation=self.output_activation, kernel_regularizer=weight_reg)(lstm)

        return Model(inputs=(inputs, ops), outputs=score)

    def _build_tf_dataset(self, cell_specs: 'list[list]', rewards: 'list[float]' = None, batch_size: int = 8, use_data_augmentation: bool = True,
                          validation_split: bool = True, shuffle: bool = True):
        return build_temporal_series_dataset_2i(self.search_space, cell_specs, rewards, batch_size, validation_split, use_data_augmentation, shuffle)