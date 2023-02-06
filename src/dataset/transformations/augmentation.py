import albumentations as alb


def get_image_classification_augmentations(use_cutout: bool):
    '''
    Keras model that can be used in both CPU or GPU for data augmentation.
    Follow similar augmentation techniques used in other papers, which usually are:

    - horizontal flip

    - 4px translate on both height and width [fill=reflect] (sometimes upscale to 40x40, with random crop to original 32x32)

    - whitening (not always used, here it's not performed)
    '''
    tx = [
        alb.HorizontalFlip(),
        alb.Affine(translate_percent=0.125)
    ]

    # TODO: implemented as before, but can also accept ratio and perform multiple holes! Experiment with these parameters...
    if use_cutout:
        tx.append(alb.CoarseDropout(max_holes=1, max_height=8, max_width=8))

    return alb.Compose(tx)


def get_image_segmentation_augmentations(crop_size: 'tuple[int, int]'):
    h, w = crop_size
    return alb.Compose([
        alb.HorizontalFlip(),
        alb.RandomCrop(height=h, width=w)
    ])
