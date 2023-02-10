import argparse
import os

import pandas as pd

from utils.experiments_summary import SearchInfo, write_search_infos_csv, FinalTrainingInfo, write_final_training_infos_csv


def get_and_write_search_info(log_path: str):
    ''' Get search info and also write a single line CSV in the experiment folder. '''
    search_info = SearchInfo.from_log_folder(log_path)
    write_search_infos_csv(os.path.join(log_path, 'csv', 'search_results.csv'), [search_info])

    return search_info


def get_and_write_final_training_info(log_path: str):
    ''' Get final training info and also write a single line CSV in the experiment folder. '''
    final_training_info = FinalTrainingInfo.from_log_folder(log_path)
    write_final_training_infos_csv(os.path.join(log_path, 'final_model_training', 'results.csv'), [final_training_info])

    return final_training_info


def save_multiple_experiments_info(log_paths: 'list[str]', summary_files_save_location: str, summary_files_prefix: str = 'exp'):
    ''' Extract info from multiple experiments, writing a CSV per experiment and one summarizing all of them for comparisons. '''
    search_infos = [get_and_write_search_info(uni_path) for uni_path in log_paths]
    final_training_infos = [get_and_write_final_training_info(uni_path) for uni_path in log_paths]

    write_search_infos_csv(os.path.join(summary_files_save_location, f'{summary_files_prefix}_search.csv'), search_infos)
    write_final_training_infos_csv(os.path.join(summary_files_save_location, f'{summary_files_prefix}_final_training.csv'), final_training_infos)


def save_tsc_aggregated_info(summary_files_save_location: str):
    ''' Write csv aggregating univariate and multivariate results. '''
    uni_search_df = pd.read_csv(os.path.join(summary_files_save_location, 'univariate_search.csv'), index_col='Dataset')
    multi_search_df = pd.read_csv(os.path.join(summary_files_save_location, 'multivariate_search.csv'), index_col='Dataset')
    pd.concat([uni_search_df, multi_search_df]).to_csv(os.path.join(summary_files_save_location, 'all_search.csv'), float_format='%.3f')

    uni_final_df = pd.read_csv(os.path.join(summary_files_save_location, 'univariate_final_training.csv'), index_col='Dataset')
    multi_final_df = pd.read_csv(os.path.join(summary_files_save_location, 'multivariate_final_training.csv'), index_col='Dataset')
    pd.concat([uni_final_df, multi_final_df]).to_csv(os.path.join(summary_files_save_location, 'all_final_training.csv'), float_format='%.3f')


def execute(p: 'list[str]', tsca: bool = False):
    ''' Refer to argparse help for more information about these arguments. '''
    if tsca:
        if len(p) != 1:
            raise AttributeError('Provide only the TSC archives root folder when using the --tsca option')

        root_folder = p[0]
        # noinspection PyTypeChecker
        univar_paths = sorted(os.scandir(os.path.join(root_folder, 'univariate')), key=lambda e: e.name)    # type: list[os.DirEntry]
        # noinspection PyTypeChecker
        multivar_paths = sorted(os.scandir(os.path.join(root_folder, 'multivariate')), key=lambda e: e.name)  # type: list[os.DirEntry]
        univariate_paths = [f.path for f in univar_paths if f.is_dir()]
        multivariate_paths = [f.path for f in multivar_paths if f.is_dir()]

        save_multiple_experiments_info(univariate_paths, root_folder, summary_files_prefix='univariate')
        save_multiple_experiments_info(multivariate_paths, root_folder, summary_files_prefix='multivariate')
        save_tsc_aggregated_info(root_folder)
    else:
        save_multiple_experiments_info(p, p[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-p', metavar='FOLDERS', nargs='+', type=str, help="log folders", required=True)
    parser.add_argument('--tsca', help="process a folder composed of multiple experiments on UCR/UEA archives", action="store_true")
    args = parser.parse_args()

    execute(**vars(args))
