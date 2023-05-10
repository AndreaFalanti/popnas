import argparse
import os

import log_service
from models.generators.factory import get_model_generator_class_for_task
from predictors.models import *
from search_space import SearchSpace
from utils.config_utils import retrieve_search_config
from utils.feature_analysis import generate_dataset_correlation_heatmap
from utils.nn_utils import initialize_train_strategy, remove_annoying_tensorflow_messages

remove_annoying_tensorflow_messages()


def setup_folders(log_path: str):
    '''
    Create folder for storing test results. If the folder already exists, keep it and previous results.
    Note: predictors will override results if they have the same name of an already existing folder.

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


# TODO: a bit messy and also "hardcodes" the predictors to test, but since it is just a utility external to the main workflow,
#  it would be a poor time investment to define a sort of grammar and configuration to refine this procedure.
def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-p', metavar='FOLDER', type=str, help="log folder", required=True)
    args = parser.parse_args()

    t_strategy = initialize_train_strategy('GPU', False)
    run_config = retrieve_search_config(args.p)

    log_path = setup_folders(args.p)
    logger = create_logger(__name__, log_path)

    search_space = SearchSpace(run_config.search_space)

    metric = run_config.search_strategy.score_metric
    logger.info('The score metric targeted is: %s', metric)
    model_gen_class = get_model_generator_class_for_task(run_config.dataset.type)
    keras_metrics = model_gen_class.get_results_processor_class().keras_metrics_considered()
    score_metric = next(m for m in keras_metrics if m.name == metric)

    csv_path = os.path.join(args.p, 'csv')
    amllibrary_config_path = os.path.join(os.getcwd(), 'configs', 'amllibrary.ini')
    training_acc_csv_path = os.path.join(csv_path, 'training_score.csv')
    catboost_col_desc_file_path = os.path.join(csv_path, 'column_desc_acc.csv')
    nn_training_data_path = os.path.join(csv_path, 'training_results.csv')
    nn_y_col = score_metric.results_csv_column
    nn_y_domain = (0, 1)

    # compute dataset correlation
    generate_dataset_correlation_heatmap(training_acc_csv_path, log_path, save_name='dataset_corr_heatmap.png')
    logger.info('Dataset correlation heatmap generated')

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
