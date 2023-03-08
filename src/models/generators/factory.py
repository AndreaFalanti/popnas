from typing import Optional, Type

from tensorflow.keras import Sequential

from models.generators import *
from utils.config_dataclasses import CnnHpConfig, ArchitectureParametersConfig


def model_generator_factory(task: str, cnn_hp: CnnHpConfig, arc_params: ArchitectureParametersConfig, training_steps_per_epoch: int,
                            output_classes_count: int, input_shape: 'tuple[int, ...]', data_augmentation_model: Optional[Sequential] = None,
                            preprocessing_model: Optional[Sequential] = None, save_weights: bool = False) -> BaseModelGenerator:
    ''' Return the right model generator, based on the task type. '''
    if task == 'image_classification' or task == 'time_series_classification':
        return ClassificationModelGenerator(cnn_hp, arc_params, training_steps_per_epoch, output_classes_count, input_shape,
                                            data_augmentation_model, preprocessing_model, save_weights)
    elif task == 'image_segmentation':
        return SegmentationFixedDecoderModelGenerator(cnn_hp, arc_params, training_steps_per_epoch, output_classes_count, input_shape,
                                                      data_augmentation_model, preprocessing_model, save_weights)
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
