import argparse
import os
from enum import Enum

import numpy as np
from PIL import Image


class ConversionType(Enum):
    RGB = 'R'
    # GRAYSCALE = 'G'
    TENSOR = 'T'

    @staticmethod
    def values():
        return [c.value for c in ConversionType]


def extract_palette(folder: str):
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
    with open(os.path.join(save_folder, 'classes.txt'), 'w') as f:
        f.writelines(f'{class_index}: {color_value}\n' for class_index, color_value in enumerate(palette))


def convert_from_rgb_to_tensor(folder: str, colors: set):
    mask_files = [f for f in os.scandir(folder) if f.is_file()]  # type: list[os.DirEntry]
    # store all masks in a dictionary, which will be saved into a compressed npz file.
    # the key is equal to the image name, while the value is the numpy array.
    np_masks_dict = {}

    for mask in mask_files:
        with Image.open(mask.path) as img:
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

            # remove the extension, since it can differ between images and masks
            save_key = mask.name.split('.')[0]
            np_masks_dict.update({save_key: converted_array})

    save_folder = os.path.dirname(folder)
    np.savez_compressed(os.path.join(save_folder, 'masks.npz'), **np_masks_dict)


def main(mask_paths: 'list[str]', conversion_mode: str):
    conversion_mode = conversion_mode.upper()
    init_type, target_type = conversion_mode.split(':')

    if init_type not in ConversionType.values() or target_type not in ConversionType.values():
        raise AttributeError('Invalid conversion type')
    if init_type == target_type:
        raise AttributeError('Conversion types are set to same value...')

    # extract a color palette from the first folder, all folders are supposed to have the same palette.
    if init_type == ConversionType.RGB.value:
        colors = extract_palette(mask_paths[0])

        if target_type == ConversionType.TENSOR.value:
            for mask_path in mask_paths:
                save_palette_info(colors, os.path.dirname(mask_path))
                convert_from_rgb_to_tensor(mask_path, colors)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-p', metavar='FOLDER_PATHS', nargs='+', type=str, help="path to log folders with masks to convert", required=True)
    parser.add_argument('-mode', metavar='CONVERSION_MODE', help='2 letters separated by colon, first is original format, second is target format. '
                                                                 'Supported values: R for RGB, T for tensor.', required=True)
    args = parser.parse_args()

    main(args.p, args.mode)
