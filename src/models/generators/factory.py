from typing import Type

from tensorflow.keras import Sequential

from models.generators import *
from utils.config_dataclasses import *


def model_generator_factory(ds_config: DatasetConfig, train_hp: TrainingHyperparametersConfig, arc_hp: ArchitectureHyperparametersConfig,
                            training_steps_per_epoch: int, output_classes_count: int, input_shape: 'tuple[int, ...]',
                            data_augmentation_model: Optional[Sequential] = None, preprocessing_model: Optional[Sequential] = None,
                            save_weights: bool = False) -> BaseModelGenerator:
    ''' Return the right model generator, based on the task type. '''
    task = ds_config.type

    if task == 'image_classification' or task == 'time_series_classification':
        return ClassificationModelGenerator(train_hp, arc_hp, training_steps_per_epoch, output_classes_count, input_shape,
                                            data_augmentation_model, preprocessing_model, save_weights)
    elif task == 'image_segmentation':
        return SegmentationFixedDecoderModelGenerator(train_hp, arc_hp, training_steps_per_epoch, output_classes_count, input_shape,
                                                      data_augmentation_model, preprocessing_model, save_weights,
                                                      ignore_class=ds_config.ignore_class)
    else:
        raise ValueError('Dataset task type is not supported by POPNAS or invalid')


def get_model_generator_class_for_task(task: str) -> Type[BaseModelGenerator]:
    '''
    Pseudo-factory method to just return the class of the model generator, without instantiating it.
    Useful to access TrainingResults and its metrics in some scripts where the actual model generator is not required.
    '''
    if task == 'image_classification' or task == 'time_series_classification':
        return ClassificationModelGenerator
    elif task == 'image_segmentation':
        return SegmentationFixedDecoderModelGenerator
    else:
        raise ValueError('Dataset task type is not supported by POPNAS or invalid')
