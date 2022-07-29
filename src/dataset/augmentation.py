from tensorflow.keras import layers
from tensorflow.python.keras import Sequential


def get_image_data_augmentation_model():
    '''
    Keras model that can be used in both CPU or GPU for data augmentation.
    Follow similar augmentation techniques used in other papers, which usually are:
    - horizontal flip
    - 4px translate on both height and width [fill=reflect] (sometimes upscale to 40x40, with random crop to original 32x32)
    - whitening (not always used, here it's not performed)
    '''
    return Sequential([
        layers.experimental.preprocessing.RandomFlip('horizontal'),
        # layers.experimental.preprocessing.RandomRotation(20/360),   # 20 degrees range
        # layers.experimental.preprocessing.RandomZoom(height_factor=0.1, width_factor=0.1),
        layers.experimental.preprocessing.RandomTranslation(height_factor=0.125, width_factor=0.125)
    ], name='data_augmentation')
