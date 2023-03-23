import spektral.layers as g_layers
from spektral.transforms import GCNFilter
from tensorflow.keras import activations, regularizers, layers, Model

from .spektral_predictor import SpektralPredictor


class GCNPredictor(SpektralPredictor):
    def _setup_data_transforms(self) -> list:
        return [GCNFilter()]

    def _get_default_hp_config(self):
        return dict(super()._get_default_hp_config(), **{
            'f1': 30,
            'f2': 60,
            'f3': 120,
            'dense_units': 100
        })

    def _get_hp_search_space(self):
        hp.Float('lr', 1e-4, 1e-2, sampling='log')
        hp.Int('f1', 20, 150, step=10, sampling='linear')
        hp.Int('f2', 20, 150, step=10, sampling='linear')
        hp.Int('f3', 20, 150, step=10, sampling='linear')
        hp.Int('dense_units', 20, 150, step=10, sampling='linear')

        return hp

    def _build_model(self, config: dict):
        weight_reg = regularizers.l2(config['wr']) if config['use_wr'] else None

        # None refers to num_nodes, since they can vary between graphs
        node_features = layers.Input(shape=(None, self.num_node_features))
        adj_matrix = layers.Input(shape=(None, None))

        gconv1 = g_layers.GCNConv(config['f1'], activation=activations.swish, kernel_regularizer=weight_reg)([node_features, adj_matrix])
        gconv2 = g_layers.GCNConv(config['f2'], activation=activations.swish, kernel_regularizer=weight_reg)([gconv1, adj_matrix])
        gconv3 = g_layers.GCNConv(config['f3'], activation=activations.swish, kernel_regularizer=weight_reg)([gconv2, adj_matrix])
        # global_feat = ExtractLastNodeFeatures()(gconv3)
        global_feat = g_layers.GlobalAvgPool()(gconv3)
        dense = layers.Dense(config['dense_units'], activation=activations.swish, kernel_regularizer=weight_reg)(global_feat)
        score = layers.Dense(1, kernel_regularizer=weight_reg)(dense)
        out = layers.Activation(self.output_activation)(score)

        return Model(inputs=[node_features, adj_matrix], outputs=out)
