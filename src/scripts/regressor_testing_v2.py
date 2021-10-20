import argparse
import math
import os

from encoder import StateSpace
import log_service
from predictors import *
from utils.func_utils import create_empty_folder

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # disable Tensorflow info messages

def setup_folders(log_path):
    regressors_test_path = os.path.join(log_path, 'regressors_test')
    create_empty_folder(regressors_test_path)

    return regressors_test_path


def create_logger(name, log_path):
    log_service.set_log_path(log_path)
    return log_service.get_logger(name)


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-p', metavar='FOLDER', type=str, help="log folder", required=True)
    args = parser.parse_args()

    log_path = setup_folders(args.p)
    logger = create_logger(__name__, log_path)

    csv_path = os.path.join(args.p, 'csv')
    amllibrary_config_path = os.path.join(os.getcwd(), 'configs', 'regressors_hyperopt.ini')
    training_time_csv_path = os.path.join(csv_path, 'training_time.csv')
    catboost_col_desc_file_path = os.path.join(csv_path, 'column_desc_time.csv')
    nn_training_data_path = os.path.join(csv_path, 'training_results.csv')

    # TODO: get these info from file from keeping consistency with choices of run tested.
    #  Right now the operators set in runs executed is always this one, but could change in future.
    operators = ['identity', '3x3 dconv', '5x5 dconv', '7x7 dconv', '1x7-7x1 conv', '3x3 conv', '3x3 maxpool', '3x3 avgpool']
    state_space = StateSpace(B=5, operators=operators, input_lookback_depth=-2)

    predictors_to_test = [
        AMLLibraryPredictor(amllibrary_config_path, ['NNLS'], logger, log_path),
        AMLLibraryPredictor(amllibrary_config_path, ['LRRidge'], logger, log_path),
        # AMLLibraryPredictor(amllibrary_config_path, ['SVR'], logger, log_path),
        # AMLLibraryPredictor(amllibrary_config_path, ['XGBoost'], logger, log_path),
        CatBoostPredictor(catboost_col_desc_file_path, logger, log_path),
        LSTMPredictor(state_space, 'training time(seconds)', (0, math.inf), logger, log_path,
                      lr=0.01, weight_reg=1e-6, embedding_dim=20, lstm_cells=100),
    ]  # type: 'list[Predictor]'

    for p in predictors_to_test:
        # TODO: formalize a method to choose correct file
        dataset_path = nn_training_data_path if isinstance(p, LSTMPredictor) else training_time_csv_path

        logger.info('%s', '*' * 36 + f' Testing predictor "{p.name}" ' + '*' * 36)
        p.perform_prediction_test(dataset_path, 'time')
        logger.info('%s', '*' * 100)

    logger.info('Script completed successfully')


if __name__ == '__main__':
    main()
