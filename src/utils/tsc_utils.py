import os
from typing import Any


class TSCDatasetMetadata:
    def __init__(self, json_metadata: 'dict[str, Any]') -> None:
        self.name = json_metadata['Dataset']
        self.train_size = int(json_metadata['Train_size'])
        self.test_size = int(json_metadata['Test_size'])
        self.timesteps = int(json_metadata['Length'])
        self.classes = int(json_metadata['Number_of_classes'])
        self.multivariate = json_metadata['Multivariate_flag'] == 1


def get_split_files_paths(root_folder: str, ds_name: str):
    train_path = [filename for filename in os.listdir(os.path.join(root_folder, ds_name)) if filename.lower().endswith('train.ts')][0]
    test_path = [filename for filename in os.listdir(os.path.join(root_folder, ds_name)) if filename.lower().endswith('test.ts')][0]

    return train_path, test_path
