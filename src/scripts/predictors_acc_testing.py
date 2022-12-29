import argparse
import os

import log_service
from predictors.models import *
from utils.feature_analysis import generate_dataset_correlation_heatmap
from utils.feature_utils import metrics_fields_dict
from utils.func_utils import instantiate_search_space_from_logs
from utils.nn_utils import initialize_train_strategy, remove_annoying_tensorflow_messages

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
    test_path = os.path.join(log_path, 'pred_acc_test')
    os.makedirs(test_path, exist_ok=True)

    return test_path


def create_logger(name, log_path):
    log_service.set_log_path(log_path)
    return log_service.get_logger(name)


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-p', metavar='FOLDER', type=str, help="log folder", required=True)
    args = parser.parse_args()

    t_strategy = initialize_train_strategy('GPU')

    log_path = setup_folders(args.p)
    logger = create_logger(__name__, log_path)

    metric = 'f1_score'

    csv_path = os.path.join(args.p, 'csv')
    amllibrary_config_path = os.path.join(os.getcwd(), 'configs', 'regressors_hyperopt.ini')
    training_acc_csv_path = os.path.join(csv_path, 'training_score.csv')
    catboost_col_desc_file_path = os.path.join(csv_path, 'column_desc_acc.csv')
    nn_training_data_path = os.path.join(csv_path, 'training_results.csv')
    nn_y_col = metrics_fields_dict[metric].real_column
    nn_y_domain = (0, 1)

    # compute dataset correlation
    generate_dataset_correlation_heatmap(training_acc_csv_path, log_path, save_name='dataset_corr_heatmap.png')
    logger.info('Dataset correlation heatmap generated')

    search_space = instantiate_search_space_from_logs(args.p)

    predictors_to_test = [
        # AMLLibraryPredictor(amllibrary_config_path, ['NNLS'], logger, log_path),
        # AMLLibraryPredictor(amllibrary_config_path, ['LRRidge'], logger, log_path),
        # AMLLibraryPredictor(amllibrary_config_path, ['SVR'], logger, log_path),
        # AMLLibraryPredictor(amllibrary_config_path, ['XGBoost'], logger, log_path),
        # CatBoostPredictor(catboost_col_desc_file_path, logger, log_path),
        # AttentionRNNPredictor(search_space, nn_y_col, nn_y_domain, t_strategy, logger, log_path, hp_tuning=False),
        # RNNPredictor(search_space, nn_y_col, nn_y_domain, t_strategy, logger, log_path, hp_tuning=False),
        # Conv1DPredictor(search_space, nn_y_col, nn_y_domain, t_strategy, logger, log_path, hp_tuning=False),
        # Conv1D1IPredictor(search_space, nn_y_col, nn_y_domain, t_strategy, logger, log_path, hp_tuning=False),
        GCNPredictor(search_space, nn_y_col, nn_y_domain, t_strategy, logger, log_path, hp_tuning=False)
    ]  # type: 'list[Predictor]'

    ensemble_units = 1   # set it to 1 to use a single model (no ensemble)
    for p in predictors_to_test:
        dataset_path = nn_training_data_path if isinstance(p, KerasPredictor) else training_acc_csv_path

        logger.info('%s', '*' * 36 + f' Testing predictor "{p.name}" ' + '*' * 36)
        p.perform_prediction_test(dataset_path, metric, ensemble_count=ensemble_units)
        logger.info('%s', '*' * 100)

    logger.info('Script completed successfully')


if __name__ == '__main__':
    main()
