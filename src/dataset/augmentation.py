import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import Sequential
from tensorflow.keras import layers

SEED = 1234


def get_image_data_augmentation_model():
    '''
    Keras model that can be used in both CPU or GPU for data augmentation.
    Follow similar augmentation techniques used in other papers, which usually are:

    - horizontal flip

    - 4px translate on both height and width [fill=reflect] (sometimes upscale to 40x40, with random crop to original 32x32)

    - whitening (not always used, here it's not performed)
    '''
    return Sequential([
        layers.RandomFlip('horizontal'),
        # layers.RandomRotation(20/360),   # 20 degrees range
        # layers.RandomZoom(height_factor=0.1, width_factor=0.1),
        layers.RandomTranslation(height_factor=0.125, width_factor=0.125)
    ], name='data_augmentation')


def get_image_classification_tf_data_aug():
    def ds_mappable_f(x, y):
        return tfa.image.random_cutout(x, mask_size=(8, 8), constant_values=0), y

    return ds_mappable_f


def get_image_segmentation_tf_data_aug(crop_size: 'tuple[int, int]', image_channels: int, num_classes):
    def ds_mappable_f(x, y):
        x, y = tf.image.random_crop(x, crop_size + (image_channels,), seed=SEED), tf.image.random_crop(y, crop_size + (num_classes,), seed=SEED)
        x, y = tf.image.random_flip_left_right(x, seed=SEED), tf.image.random_flip_left_right(y, seed=SEED)
        return x, y

    return ds_mappable_f
