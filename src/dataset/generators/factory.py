from dataset.generators import *
from utils.config_dataclasses import DatasetConfig, OthersConfig


def dataset_generator_factory(ds_config: DatasetConfig, others_config: OthersConfig) -> BaseDatasetGenerator:
    '''
    Return the right dataset generator, based on the task type.
    '''
    task_type = ds_config.type
    optimize_for_xla_compilation = others_config.enable_XLA_compilation or others_config.train_strategy == 'TPU'

    if task_type == 'image_classification':
        return ImageClassificationDatasetGenerator(ds_config, optimize_for_xla_compilation)
    elif task_type == 'time_series_classification':
        return TimeSeriesClassificationDatasetGenerator(ds_config, optimize_for_xla_compilation)
    elif task_type == 'image_segmentation':
        return ImageSegmentationDatasetGenerator(ds_config, optimize_for_xla_compilation)
    else:
        raise ValueError('Dataset task type is not supported by POPNAS or invalid')
