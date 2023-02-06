from typing import Optional

import albumentations as alb
import tensorflow as tf


def labels_to_one_hot(num_classes: int):
    def one_hot(y):
        return tf.one_hot(y, num_classes)

    return one_hot


def preprocess_image(resize_dim: 'Optional[tuple[int, int]]', image_to_float: bool):
    preprocessing_tx = []
    if resize_dim is not None:
        h, w = resize_dim
        preprocessing_tx.append(alb.Resize(h, w))
    if image_to_float:
        preprocessing_tx.append(alb.ToFloat())

    return alb.Compose(preprocessing_tx)
