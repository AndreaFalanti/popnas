import argparse
import math
import os

import log_service
from encoder import StateSpace
from predictors import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # disable Tensorflow info messages


def setup_folders(log_path: str):
    '''
    Create folder for storing test results. If folder already exists, keep it and previous results.
    Note: predictors will override results if have same name of an already existing folder.

    Args:
        log_path: path of run logs folder

    Returns:
        (str): path for script output
    '''
    test_path = os.path.join(log_path, 'pred_time_test')
    os.makedirs(test_path, exist_ok=True)

    return test_path


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
    nn_y_col = 'training time(seconds)'
    nn_y_domain = (0, math.inf)

    new_catboost_col_desc_file_path = os.path.join(csv_path, 'new_column_desc_time.csv')
    new_training_time_csv_path = os.path.join(csv_path, 'new_training_time.csv')

    # TODO: get these info from file from keeping consistency with choices of run tested.
    #  Right now the operators set in runs executed is always this one, but could change in future.
    operators = ['identity', '3x3 dconv', '5x5 dconv', '7x7 dconv', '1x7-7x1 conv', '3x3 conv', '3x3 maxpool', '3x3 avgpool']
    state_space = StateSpace(B=5, operators=operators, cell_stack_depth=8, input_lookback_depth=-2)

    predictors_to_test = [
        # AMLLibraryPredictor(amllibrary_config_path, ['NNLS'], logger, log_path, name='aMLLibrary_NNLS (new features)'),
        # AMLLibraryPredictor(amllibrary_config_path, ['LRRidge'], logger, log_path, name='aMLLibrary_LRRidge (new features)'),
        # AMLLibraryPredictor(amllibrary_config_path, ['SVR'], logger, log_path),
        # AMLLibraryPredictor(amllibrary_config_path, ['XGBoost'], logger, log_path, name='aMLLibrary_XGBoost (new features)'),
        # CatBoostPredictor(catboost_col_desc_file_path, logger, log_path),
        # LSTMPredictor(state_space, nn_y_col, nn_y_domain, logger, log_path,
        #               lr=0.01, weight_reg=1e-6, embedding_dim=20, rnn_cells=100),
        # Conv1DPredictor(state_space, nn_y_col, nn_y_domain, logger, log_path, lr=0.005, weight_reg=0, epochs=20),
        # GRUPredictor(state_space, nn_y_col, nn_y_domain, logger, log_path,
        #              epochs=20, lr=0.01, weight_reg=1e-6, embedding_dim=20, rnn_cells=100),
        CatBoostPredictor(new_catboost_col_desc_file_path, logger, log_path, name='CatBoost (new features)'),
    ]  # type: 'list[Predictor]'

    for p in predictors_to_test:
        dataset_path = nn_training_data_path if isinstance(p, NNPredictor) else new_training_time_csv_path

        logger.info('%s', '*' * 36 + f' Testing predictor "{p.name}" ' + '*' * 36)
        p.perform_prediction_test(dataset_path, 'time')
        logger.info('%s', '*' * 100)

    logger.info('Script completed successfully')


if __name__ == '__main__':
    main()
