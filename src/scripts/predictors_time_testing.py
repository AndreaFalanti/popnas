import argparse
import math
import os

import log_service
from predictors.models import *
from search_space import SearchSpace
from utils.config_utils import retrieve_search_config
from utils.feature_analysis import generate_dataset_correlation_heatmap
from utils.nn_utils import remove_annoying_tensorflow_messages

remove_annoying_tensorflow_messages()


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


# TODO: a bit messy and also "hardcodes" the predictors to test, but since it is just a utility external to the main workflow,
#  it would be a poor time investment to define a sort of grammar and configuration to refine this procedure.
def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-p', metavar='FOLDER', type=str, help="log folder", required=True)
    args = parser.parse_args()

    run_config = retrieve_search_config(args.p)

    log_path = setup_folders(args.p)
    logger = create_logger(__name__, log_path)

    csv_path = os.path.join(args.p, 'csv')
    amllibrary_config_path = os.path.join(os.getcwd(), 'configs', 'regressors_hyperopt.ini')
    training_time_csv_path = os.path.join(csv_path, 'training_time.csv')
    catboost_col_desc_file_path = os.path.join(csv_path, 'column_desc_time.csv')
    nn_training_data_path = os.path.join(csv_path, 'training_results.csv')
    nn_y_col = 'training time(seconds)'
    nn_y_domain = (0, math.inf)

    # compute dataset correlation
    generate_dataset_correlation_heatmap(training_time_csv_path, log_path, save_name='dataset_corr_heatmap.png')
    logger.info('Dataset correlation heatmap generated')

    search_space = SearchSpace(run_config.search_space)

    predictors_to_test = [
        AMLLibraryPredictor(amllibrary_config_path, ['NNLS'], logger, log_path),
        AMLLibraryPredictor(amllibrary_config_path, ['LRRidge'], logger, log_path),
        AMLLibraryPredictor(amllibrary_config_path, ['SVR'], logger, log_path),
        AMLLibraryPredictor(amllibrary_config_path, ['XGBoost'], logger, log_path),
        AMLLibraryPredictor(amllibrary_config_path, ['NNLS', 'LRRidge', 'SVR', 'XGBoost'], logger, log_path, name='aMLLibrary_ALL'),
        # RNNPredictor(search_space, nn_y_col, nn_y_domain, logger, log_path, hp_tuning=False),
        # Conv1DPredictor(search_space, nn_y_col, nn_y_domain, logger, log_path, hp_tuning=False),
        # Conv1D1IPredictor(search_space, nn_y_col, nn_y_domain, logger, log_path, hp_tuning=False),
        # CatBoostPredictor(catboost_col_desc_file_path, logger, log_path, use_random_search=False),
        # CatBoostPredictor(catboost_col_desc_file_path, logger, log_path, use_random_search=True, task_type='GPU', name='CatBoost_GPU_NEW'),
        # LGBMPredictor(logger, log_path, drop_feature_names=['exploration'], use_random_search=True)
    ]  # type: 'list[Predictor]'

    for p in predictors_to_test:
        dataset_path = nn_training_data_path if isinstance(p, KerasPredictor) else training_time_csv_path

        logger.info('%s', '*' * 36 + f' Testing predictor "{p.name}" ' + '*' * 36)
        p.perform_prediction_test(dataset_path, 'time')
        logger.info('%s', '*' * 100)

    logger.info('Script completed successfully')


if __name__ == '__main__':
    main()
