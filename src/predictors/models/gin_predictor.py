import spektral.layers as g_layers
from tensorflow.keras import activations, regularizers, layers, Model

from .spektral_predictor import SpektralPredictor


class GINPredictor(SpektralPredictor):
    def _get_default_hp_config(self):
        return dict(super()._get_default_hp_config(), **{
            'lr': 1e-3,
            'wr': 1e-5,
            'f1': 20,
            'f2': 30,
            'f3': 40,
            'dense_units': 50
        })

    def _get_hp_search_space(self):
        hp = super()._get_hp_search_space()
        hp.Float('lr', 1e-4, 1e-2, sampling='log')
        hp.Float('wr', 1e-7, 1e-4, sampling='log')
        hp.Int('f1', 10, 100, sampling='linear')
        hp.Int('f2', 10, 100, sampling='linear')
        hp.Int('f3', 10, 100, sampling='linear')
        hp.Int('dense_units', 10, 100, sampling='linear')

        return hp

    def _build_model(self, config: dict):
        weight_reg = regularizers.l2(config['wr']) if config['wr'] > 0 else None

        # None refers to num_nodes, since they can vary between graphs
        node_features = layers.Input(shape=(None, self.num_node_features))
        adj_matrix = layers.Input(shape=(None, None))

        gconv1 = g_layers.GINConvBatch(config['f1'], activation=activations.swish, kernel_regularizer=weight_reg)([node_features, adj_matrix])
        gconv2 = g_layers.GINConvBatch(config['f2'], activation=activations.swish, kernel_regularizer=weight_reg)([gconv1, adj_matrix])
        gconv3 = g_layers.GINConvBatch(config['f3'], activation=activations.swish, kernel_regularizer=weight_reg)([gconv2, adj_matrix])
        # get only features of exit node (also note that einops and tf functions consider also batch, while Keras functional API (Inputs) does not)
        # global_feat = ExtractLastNodeFeatures(config['f3'])(gconv3)
        global_feat = g_layers.GlobalAvgPool()(gconv3)
        dense = layers.Dense(config['dense_units'], activation=activations.swish, kernel_regularizer=weight_reg)(global_feat)
        score = layers.Dense(1, kernel_regularizer=weight_reg)(dense)
        out = layers.Activation(self.output_activation)(score)

        return Model(inputs=[node_features, adj_matrix], outputs=out)
