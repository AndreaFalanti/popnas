import argparse
import math
import os
from enum import Enum
from typing import Optional

import numpy as np
from PIL import Image
from tqdm import tqdm


class ConversionType(Enum):
    RGB = 'R'
    # GRAYSCALE = 'G'
    TENSOR = 'T'

    @staticmethod
    def values():
        return [c.value for c in ConversionType]


def extract_palette(folder: str):
    ''' Extract all the colors used in masks. This info is needed to correctly one-hot encode the masks. '''
    # os.DirEntry
    mask_files = [f.path for f in os.scandir(folder) if f.is_file()]  # type: list[str]
    colors_set = set()
    for mask in mask_files:
        with Image.open(mask) as img:
            colors = img.getcolors()

            if len(colors[0]) == 2:
                colors_set.update([pixel_value for count, pixel_value in colors])
            else:
                raise NotImplementedError('24-bit RPG is not supported currently')

    return colors_set


def save_palette_info(palette: set, save_folder: str):
    ''' Save the masks palette into a txt files, mapping class indexes to colors. '''
    with open(os.path.join(save_folder, 'classes.txt'), 'w') as f:
        f.writelines(f'{class_index}: {color_value}\n' for class_index, color_value in enumerate(palette))


def convert_from_rgb_to_tensor(img: Image.Image, colors: set):
    ''' Convert a mask from RGB to one-hot encoded tensor (H, W, # classes). '''
    img_np = np.asarray(img)
    original_shape = img_np.shape

    # if not grayscale or 8bit png, extract only the image resolution as shape (no channels)
    if len(original_shape) == 3:
        original_shape = original_shape[:-1]

    converted_shape = original_shape + (len(colors),)

    converted_array = np.zeros(converted_shape, dtype=np.int8)
    for i, color in enumerate(colors):
        # get a boolean (0/1) mask for each class
        converted_array[:, :, i] = np.where(img_np == color, 1, 0)

    return converted_array


def extract_palette_from_masks(mask_path: str, masks_init_type: str):
    # extract a color palette from the first folder, all folders are supposed to have the same palette.
    if masks_init_type == ConversionType.RGB.value:
        return extract_palette(mask_path)
    else:
        raise AttributeError('Not supported mask init type')


def convert_mask(mask: Image, masks_palette: set, target_type: str):
    ''' Convert masks to target type. '''
    if target_type == ConversionType.TENSOR.value:
        return convert_from_rgb_to_tensor(mask, masks_palette)
    elif target_type == ConversionType.RGB.value:
        arr = np.asarray(mask)
        return np.expand_dims(arr, axis=-1)
    else:
        raise AttributeError('Not supported conversion target type')


def central_crop_and_pad_to_squared_dim(img: Image.Image, target_dim: int, pad_constant_color: int):
    '''
    Perform a central crop plus padding of the provided image.
    The crop is performed for any dimension larger than the target dimension, while the padding is performed in the opposite case.

    Args:
        img: target image
        target_dim: resize dimension for square crops
        pad_constant_color: color to use for padding

    Returns:
        the new image with size (target_dim, target_dim)
    '''
    img_w, img_h = img.width, img.height

    # make the padding value RGB if necessary
    if img.mode == 'RGB':
        pad_constant_color = (pad_constant_color, pad_constant_color, pad_constant_color)

    # central crop, if necessary
    crop_w = max(img_w - target_dim, 0)
    crop_h = max(img_h - target_dim, 0)
    if crop_w > 0 or crop_h > 0:
        # compute crop box coordinates
        box_left = math.ceil(crop_w / 2)
        box_right = math.floor(img_w - crop_w / 2)
        box_top = math.ceil(crop_h / 2)
        box_bottom = math.ceil(img_h - crop_h / 2)
        img = img.crop(box=(box_left, box_top, box_right, box_bottom))

    pad_w = max(target_dim - img_w, 0)
    pad_h = max(target_dim - img_h, 0)
    if pad_w > 0 or pad_h > 0:
        paste_left = pad_w // 2
        paste_top = pad_h // 2

        padding_base_img = Image.new(img.mode, size=(target_dim, target_dim), color=pad_constant_color)
        padding_base_img.paste(img, box=(paste_left, paste_top))
        img = padding_base_img

    return img


def main(ds_folders: 'list[str]', masks_conversion_mode: str, fixed_resize_dim: Optional[int] = None,
         resize_min_axis: Optional[int] = None, save_as_ragged: bool = False):
    '''
    Generate a segmentation dataset in a standardized .npz format, easily readable by POPNAS.
    Each split should be placed in a folder, with subfolders for RGB images and masks.

    Args:
        ds_folders: a folder for each dataset split.
         It should contain two subfolders: one named 'images', containing RGB images used as samples,
         and one named 'masks', with PNG segmentation masks.
        masks_conversion_mode: conversion mode for masks
        fixed_resize_dim: optional parameter to resize images to specific squared dimension.
        resize_min_axis: optional parameter to resize the smallest axis of images and masks to the provided value, preserving the aspect ratio.
          Ignored if fixed_resize is provided.
        save_as_ragged: save the numpy array with object type, if it is ragged.
    '''
    first_split_masks_folder = os.path.join(ds_folders[0], 'masks')

    conversion_mode = masks_conversion_mode.upper()
    init_type, target_type = conversion_mode.split(':')

    if init_type not in ConversionType.values() or target_type not in ConversionType.values():
        raise AttributeError('Invalid conversion type')
    if init_type == target_type:
        print('Conversion types are set to the same value, masks will not be converted')

    if fixed_resize_dim is not None and save_as_ragged:
        save_as_ragged = False
        print('Fixed dimension is provided, so the dataset will not be saved as ragged (it would just be a drawback, flag is overriden)')

    mask_colours = extract_palette_from_masks(first_split_masks_folder, init_type)
    print('Masks palette extracted successfully!')

    for ds_folder in ds_folders:
        print(f'Processing folder: {ds_folder}')
        image_folder = os.path.join(ds_folder, 'images')
        masks_folder = os.path.join(ds_folder, 'masks')
        save_palette_info(mask_colours, ds_folder)

        # lists that will contain the numpy arrays for samples and masks
        X, Y = [], []

        image_files = [f for f in os.scandir(image_folder) if f.is_file()]  # type: list[os.DirEntry]
        for img_entry in tqdm(image_files):
            new_size = None

            with Image.open(img_entry.path) as img:
                if fixed_resize_dim is not None:
                    img = central_crop_and_pad_to_squared_dim(img, fixed_resize_dim, pad_constant_color=0)
                elif resize_min_axis is not None:
                    min_axis_len = min(img.height, img.width)
                    ratio = resize_min_axis / min_axis_len
                    new_size = (math.floor(img.width * ratio), math.floor(img.height * ratio))

                    img = img.resize(new_size, resample=Image.BICUBIC)

                # append numpy array of RGB image
                X.append(np.asarray(img))

            fname_without_ext = img_entry.name.split('.')[0]
            with Image.open(os.path.join(masks_folder, fname_without_ext + '.png')) as img:
                if fixed_resize_dim is not None:
                    img = central_crop_and_pad_to_squared_dim(img, fixed_resize_dim, pad_constant_color=255)
                elif new_size is not None:
                    img = img.resize(new_size, resample=Image.NEAREST)

                np_mask = convert_mask(img, mask_colours, target_type)
                Y.append(np_mask)

        print('Saving NPZ archive...')
        if fixed_resize_dim:
            name_suffix = f'{fixed_resize_dim}f'
        else:
            name_suffix = '' if resize_min_axis is None else str(resize_min_axis)
        save_name = f'{os.path.basename(ds_folder)}{name_suffix}'

        # use object as dtype if arrays are ragged
        if save_as_ragged:
            np.savez_compressed(os.path.join(ds_folder, save_name), x=np.asarray(X, dtype=object), y=np.asarray(Y, dtype=object))
        else:
            np.savez_compressed(os.path.join(ds_folder, save_name), x=np.asarray(X), y=np.asarray(Y))
        print('NPZ archive saved successfully!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-p', metavar='FOLDER_PATHS', nargs='+', type=str, help="path to dataset splits to convert", required=True)
    parser.add_argument('-m', metavar='MASK_CONVERSION_MODE', help='2 letters separated by colon, first is original format, second is target format. '
                                                                   'Supported values: R for RGB, T for tensor.', required=True)
    parser.add_argument('-fr', metavar='FIXED_RESIZE_DIM', type=int, help="the target size for resizing images (squares)", default=None)
    parser.add_argument('-r', metavar='RESIZE_MIN_AXIS', type=int, help="the target size for smallest axis of the image", default=None)
    parser.add_argument('--ragged', action='store_true', help="use it if the images have different sizes", default=False)
    args = parser.parse_args()

    main(args.p, args.m, args.fr, args.r, args.ragged)
