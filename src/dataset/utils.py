from collections import Counter

import numpy as np
import tensorflow as tf
from albumentations import Compose
from matplotlib import pyplot as plt

from dataset.generators import *


def test_data_augmentation(ds: tf.data.Dataset, data_augmentation: Compose):
    '''
    Function helpful for debugging data augmentation and making sure it's working properly.
    DON'T USE IT IN ACTUAL RUNS.

    Args:
        ds: any TF dataset where data augmentation is applied
        data_augmentation: a Compose of Albumentations transformations
    '''
    # switch to an interactive matplotlib backend
    plt.switch_backend('TkAgg')

    # get a batch
    images, labels = next(iter(ds))

    # display 9 transformation of the first 3 images taken from the first training batch
    for j in range(3):
        image = images[j]
        plt.imshow(image)
        plt.show()

        for i in range(9):
            transformed = data_augmentation(image=image.numpy())
            _ = plt.subplot(3, 3, i + 1)
            plt.imshow(transformed['image'])
            plt.axis('off')

        plt.show()
    print('Data augmentation debug shown')


def get_dataset_labels_distribution(ds: tf.data.Dataset) -> 'dict[int, float]':
    ''' Returns the percentage of samples associated to each label of the dataset. '''
    label_counter = Counter()

    for (x, y) in ds:
        label_counter.update(np.argmax(y, axis=-1))

    percentages_dict = {}
    for (k, v) in label_counter.items():
        percentages_dict[k] = v / sum(label_counter.values())

    return percentages_dict


def generate_balanced_weights_for_classes(ds: tf.data.Dataset) -> 'dict[int, float]':
    ''' Compute weights to balance class importance during training with unbalanced datasets. '''
    class_percentages_dict = get_dataset_labels_distribution(ds)
    num_classes = len(class_percentages_dict.keys())

    class_weights_dict = {}
    for cl, perc in class_percentages_dict.items():
        class_weights_dict[cl] = 1 / (num_classes * perc)

    return class_weights_dict


def dataset_generator_factory(ds_config: dict, enable_tpu_tricks: bool = False) -> BaseDatasetGenerator:
    '''
    Return the right dataset generator, based on task type.
    '''
    task_type = ds_config['type']

    if task_type == 'image_classification':
        return ImageClassificationDatasetGenerator(ds_config, enable_tpu_tricks)
    elif task_type == 'time_series_classification':
        return TimeSeriesClassificationDatasetGenerator(ds_config, enable_tpu_tricks)
    elif task_type == 'image_segmentation':
        return ImageSegmentationDatasetGenerator(ds_config, enable_tpu_tricks)
    else:
        raise ValueError('Dataset task type is not supported by POPNAS or invalid')
