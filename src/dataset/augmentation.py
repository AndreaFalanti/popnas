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


def get_augmentation_random_params(x: tf.Tensor, crop_x_size: int, crop_y_size: int):
    # Get the shape of the image and mask, and the respective crop targets
    shape = tf.shape(x)
    h, w = shape[0], shape[1]

    # Randomly generate the crop start point
    crop_y_start = tf.random.uniform(shape=(), minval=0, maxval=h - crop_y_size + 1, dtype=tf.int32)
    crop_x_start = tf.random.uniform(shape=(), minval=0, maxval=w - crop_x_size + 1, dtype=tf.int32)

    # Compute if the random horizontal flip will be applied or not (50%)
    use_horizontal_flip = tf.random.uniform(()) > 0.5

    return crop_x_start, crop_y_start, use_horizontal_flip


def get_image_segmentation_tf_data_aug_xy(crop_size: 'tuple[int, int]'):
    @tf.function
    def ds_mappable_f(x, y):
        crop_x_size, crop_y_size = crop_size
        crop_x_start, crop_y_start, use_horizontal_flip = get_augmentation_random_params(x, crop_x_size, crop_y_size)

        x = tf.image.crop_to_bounding_box(x, crop_y_start, crop_x_start, crop_y_size, crop_x_size)
        y = tf.image.crop_to_bounding_box(y, crop_y_start, crop_x_start, crop_y_size, crop_x_size)
        if use_horizontal_flip:
            x = tf.image.flip_left_right(x)
            y = tf.image.flip_left_right(y)

        return x, y

    return ds_mappable_f


# same of above, but working on 3 elements (sample weights)
# unfortunately, there seems that tf.data.Dataset can map only function using the exact number of elements as the dataset tuples,
# it is not possible to use a single element as tuple and then iterate on that.
def get_image_segmentation_tf_data_aug_xys(crop_size: 'tuple[int, int]'):
    @tf.function
    def ds_mappable_f(x, y, s):
        crop_x_size, crop_y_size = crop_size
        crop_x_start, crop_y_start, use_horizontal_flip = get_augmentation_random_params(x, crop_x_size, crop_y_size)

        x = tf.image.crop_to_bounding_box(x, crop_y_start, crop_x_start, crop_y_size, crop_x_size)
        y = tf.image.crop_to_bounding_box(y, crop_y_start, crop_x_start, crop_y_size, crop_x_size)
        s = tf.image.crop_to_bounding_box(s, crop_y_start, crop_x_start, crop_y_size, crop_x_size)
        if use_horizontal_flip:
            x = tf.image.flip_left_right(x)
            y = tf.image.flip_left_right(y)
            s = tf.image.flip_left_right(s)

        return x, y, s

    return ds_mappable_f
