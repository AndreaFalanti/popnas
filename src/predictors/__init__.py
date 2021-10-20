import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from predictor import Predictor
from amllibrary_predictor import AMLLibraryPredictor
from catboost_predictor import CatBoostPredictor
from lstm_predictor import LSTMPredictor
