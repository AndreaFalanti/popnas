from typing import Optional

from tensorflow.keras import Sequential

from models.generators import *


def model_generator_factory(task: str, cnn_hp: dict, arc_params: dict, training_steps_per_epoch: int, output_classes_count: int,
                            input_shape: 'tuple[int, ...]', data_augmentation_model: Optional[Sequential] = None,
                            preprocessing_model: Optional[Sequential] = None, save_weights: bool = False) -> BaseModelGenerator:
    '''
    Return the right model generator, based on the task type.
    '''
    if task == 'image_classification' or task == 'time_series_classification':
        return ClassificationModelGenerator(cnn_hp, arc_params, training_steps_per_epoch, output_classes_count, input_shape,
                                            data_augmentation_model, preprocessing_model, save_weights)
    elif task == 'image_segmentation':
        raise NotImplementedError('TODO: image segmentation model generator is missing right now')
    else:
        raise ValueError('Dataset task type is not supported by POPNAS or invalid')
