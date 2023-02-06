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


def get_image_segmentation_tf_data_aug(crop_size: 'tuple[int, int]'):
    @tf.function
    def ds_mappable_f(x, y):
        # thanks ChatGPT for the idea of using tf.random.uniform instead of already made random functions!

        # Get the shape of the image and mask, and the respective crop targets
        shape = tf.shape(x)
        h, w = shape[0], shape[1]
        crop_x_size, crop_y_size = crop_size

        # Randomly generate the crop start point
        y_start = tf.random.uniform(shape=(), minval=0, maxval=h - crop_y_size + 1, dtype=tf.int32)
        x_start = tf.random.uniform(shape=(), minval=0, maxval=w - crop_x_size + 1, dtype=tf.int32)

        # Apply the crop to both the image and the mask
        x = tf.image.crop_to_bounding_box(x, y_start, x_start, crop_y_size, crop_x_size)
        y = tf.image.crop_to_bounding_box(y, y_start, x_start, crop_y_size, crop_x_size)

        if tf.random.uniform(()) > 0.5:
            x = tf.image.flip_left_right(x)
            y = tf.image.flip_left_right(y)

        # in this way, the transformations are not applied in the same way to image and masks...
        # x, y = tf.image.random_crop(x, crop_size + (image_channels,), seed=SEED), tf.image.random_crop(y, crop_size + (num_classes,), seed=SEED)
        # x, y = tf.image.random_flip_left_right(x, seed=SEED), tf.image.random_flip_left_right(y, seed=SEED)

        return x, y

    return ds_mappable_f
