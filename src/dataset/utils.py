from collections import Counter

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from dataset.augmentation import get_image_segmentation_tf_data_aug_xy


def test_data_augmentation(ds: tf.data.Dataset):
    '''
    Function helpful for debugging data augmentation and making sure it's working properly.
    DON'T USE IT IN ACTUAL RUNS.

    Args:
        ds: any TF dataset where data augmentation is applied
    '''
    # switch to an interactive matplotlib backend
    plt.switch_backend('TkAgg')

    # data_augmentation_model = get_image_data_augmentation_model()
    data_aug = get_image_segmentation_tf_data_aug_xy((112, 112))

    # get a batch
    images, labels, weights = next(iter(ds))

    # display 8 transformations of the first 3 images belonging to the first training batch
    for j in range(3):
        image = images[j]
        label = labels[j]

        plt.imshow(image)
        # plt.imshow(label[:, :, 0])
        plt.show()

        for i in range(0, 8, 2):
            # augmented_image = data_augmentation_model(image)
            augmented_image, augmented_mask = data_aug(image, label)
            _ = plt.subplot(4, 2, i + 1)
            plt.imshow(augmented_image)
            plt.axis('off')

            _ = plt.subplot(4, 2, i + 2)
            plt.imshow(augmented_mask[:, :, 0])
            plt.axis('off')

        plt.show()
    print('Data augmentation debug shown')


def get_dataset_labels_distribution(ds: tf.data.Dataset) -> 'dict[int, float]':
    ''' Returns the percentage of samples associated to each label of the dataset. '''
    label_counter = Counter()

    for x, y in ds:
        y_is_sparse_label = tf.shape(y)[-1] == 1
        # if not a sparse label, perform one-hot to int
        labels = y.numpy() if y_is_sparse_label else np.argmax(y, axis=-1)
        # flatten is required for cases where the labels are multidimensional (e.g. segmentation masks, which have a label for each image pixel)
        label_counter.update(labels.flatten())

    percentages_dict = {}
    for k, v in label_counter.items():
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


def generate_sample_weights_from_class_weights(ds: tf.data.Dataset):
    class_weights = generate_balanced_weights_for_classes(ds)

    def map_sample_weights(x: tf.Tensor, y: tf.Tensor):
        y_is_sparse_label = tf.shape(y)[-1] == 1

        # if not a sparse label, perform one-hot to int
        y_labels = tf.cast(y, dtype=tf.int32) if y_is_sparse_label else tf.argmax(y, axis=-1, output_type=tf.int32)
        # the gather function is duplicated, but leave it as it is.
        # TF autograph needs variables with different shapes to have different names, so the class weights must have different names to work!
        if y_is_sparse_label:
            sparse_class_weights_arr = [class_weights.get(i, 0.0) for i in range(max(class_weights.keys()) + 1)]
            sample_weight = tf.gather(sparse_class_weights_arr, indices=y_labels)
        else:
            class_weights_arr = [class_weights[key] for key in sorted(class_weights.keys())]
            sample_weight = tf.gather(class_weights_arr, indices=y_labels)

        return x, y, sample_weight

    return ds.map(map_sample_weights, num_parallel_calls=tf.data.AUTOTUNE)
