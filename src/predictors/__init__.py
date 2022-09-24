import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from predictor import Predictor
from amllibrary_predictor import AMLLibraryPredictor
from catboost_predictor import CatBoostPredictor
from rnn_predictor import RNNPredictor
from keras_predictor import KerasPredictor
from conv1d_predictor import Conv1DPredictor
from conv1d1i_predictor import Conv1D1IPredictor
from rnn_attention_predictor import AttentionRNNPredictor
from lgbm_predictor import LGBMPredictor
from spektral_predictor import SpektralPredictor
from gcn_predictor import GCNPredictor
from gin_predictor import GINPredictor

__all__ = ['Predictor', 'AMLLibraryPredictor', 'CatBoostPredictor', 'KerasPredictor', 'RNNPredictor',
           'Conv1DPredictor', 'Conv1D1IPredictor', 'AttentionRNNPredictor', 'LGBMPredictor',
           'SpektralPredictor', 'GCNPredictor', 'GINPredictor']
