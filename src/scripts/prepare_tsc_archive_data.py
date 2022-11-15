import argparse
import heapq
import json
import os
import pickle
from typing import Iterable

from utils.tsc_utils import TSCDatasetMetadata, get_split_files_paths


def partition_ds_iterable(elems: Iterable[TSCDatasetMetadata], n: int):
    lists = [[] for _ in range(n)]
    totals = [(0, i) for i in range(n)]
    heapq.heapify(totals)

    for ds_meta in elems:
        total, index = heapq.heappop(totals)
        ds_cost = ds_meta.train_size * ds_meta.timesteps

        lists[index].append(ds_meta)
        heapq.heappush(totals, (total + ds_cost, index))

    return lists


def check_ds_files(root_folder: str, dataset_names: 'list[str]'):
    for ds_name in dataset_names:
        try:
            get_split_files_paths(root_folder, ds_name)
        except:
            print(f'{ds_name} datasets files not found')


def main(p: str, n: int):
    ucr_path = os.path.join(p, 'univariate')
    uea_path = os.path.join(p, 'multivariate')
    json_path = os.path.join(p, 'tsc-datasets.json')

    UNIVARIATE_DATASET_NAMES = [f.name for f in os.scandir(ucr_path) if f.is_dir()]
    MULTIVARIATE_DATASET_NAMES = [f.name for f in os.scandir(uea_path) if f.is_dir()]
    ALL_DATASET_NAMES = UNIVARIATE_DATASET_NAMES + MULTIVARIATE_DATASET_NAMES

    # just check that the files are named correctly
    print(f'Checking {len(UNIVARIATE_DATASET_NAMES)} UCR datasets and {len(MULTIVARIATE_DATASET_NAMES)} UEA datasets')
    check_ds_files(ucr_path, UNIVARIATE_DATASET_NAMES)
    check_ds_files(uea_path, MULTIVARIATE_DATASET_NAMES)

    with open(json_path, 'r') as f:
        ds_metadata_json_list = json.load(f)

    ds_metadata_list = [TSCDatasetMetadata(m) for m in ds_metadata_json_list]
    ds_metadata_filtered_list = [ds_meta for ds_meta in ds_metadata_list if ds_meta.name in ALL_DATASET_NAMES]
    print(f'Missmatch between {len(ds_metadata_list) - len(ds_metadata_filtered_list)} metadata name and dataset folders')
    # discard too small datasets and datasets with samples of unequal size (timesteps are = 0 if unequal)
    ds_metadata_filtered_list = [ds_meta for ds_meta in ds_metadata_filtered_list if ds_meta.train_size >= 360 and ds_meta.timesteps > 0]

    partitions = partition_ds_iterable(ds_metadata_filtered_list, n)
    for i, partition in enumerate(partitions):
        with open(os.path.join(p, f'tsc_split_{i}.pickle'), 'wb') as f:
            pickle.dump(partition, f)

    with open(os.path.join(p, f'tsc_split_all.pickle'), 'wb') as f:
        pickle.dump(ds_metadata_filtered_list, f)

    # for debugging, uses just two dataset: one univariate, one multivariate
    first_univariate = next((ds_meta for ds_meta in ds_metadata_filtered_list if not ds_meta.multivariate), None)
    first_multivariate = next((ds_meta for ds_meta in ds_metadata_filtered_list if ds_meta.multivariate), None)
    with open(os.path.join(p, f'tsc_split_debug.pickle'), 'wb') as f:
        pickle.dump([first_univariate, first_multivariate], f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-p', metavar='ARCHIVE_PATH', type=str, help="path to UCR and UEA datasets archive", required=True)
    parser.add_argument('-n', metavar='WORKERS_NUM', type=int, help="number of workers that will be used to execute the UCR/UEA dataset",
                        required=True)
    args = parser.parse_args()

    main(**vars(args))
