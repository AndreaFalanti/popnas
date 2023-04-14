import argparse
import os

import numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt
from tensorflow.keras import models, Model

from dataset.generators.factory import dataset_generator_factory
from search_space import CellSpecification
from utils.config_utils import read_json_config
from utils.func_utils import create_empty_folder
from utils.nn_utils import initialize_train_strategy, remove_annoying_tensorflow_messages

# disable Tensorflow info and warning messages
remove_annoying_tensorflow_messages()

AUTOTUNE = tf.data.AUTOTUNE

# dict mapping class labels to RGB colors for visual representations
# it uses this palette: https://en.wikipedia.org/wiki/Help:Distinguishable_colors, with black as class 0 and white as class 255 (ignore)
cmap_dict_rgb = {
    0: [0, 0, 0],
    1: [240, 163, 255],
    2: [0, 117, 220],
    3: [153, 63, 0],
    4: [76, 0, 92],
    5: [25, 25, 25],
    6: [0, 92, 49],
    7: [43, 206, 72],
    8: [255, 204, 153],
    9: [128, 128, 128],
    10: [148, 255, 181],
    11: [143, 124, 0],
    12: [157, 204, 0],
    13: [194, 0, 136],
    14: [0, 51, 128],
    15: [255, 164, 5],
    16: [255, 168, 187],
    17: [66, 102, 0],
    18: [255, 0, 16],
    19: [94, 241, 242],
    20: [0, 153, 143],
    21: [224, 255, 102],
    22: [116, 10, 255],
    23: [153, 0, 0],
    24: [255, 255, 128],
    25: [255, 225, 0],
    26: [255, 80, 5],
    255: [255, 255, 255]
}


def get_model_cell_spec(log_folder_path: str):
    with open(os.path.join(log_folder_path, 'cell_spec.txt'), 'r') as f:
        cell_spec = f.read()

    return CellSpecification.from_str(cell_spec)


# This script can be used to evaluate the final model trained on a test set.
# It needs a saved model, which could be the one found during search or the one produced by final_training script (spec + checkpoint)
def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-p', metavar='PATH', type=str, help="path to final model folder", required=True)
    parser.add_argument('-d', metavar='DATASET_PATH', type=str, help="override dataset path", required=False)
    parser.add_argument('-n', metavar='NUM_SAMPLES', type=int, help="number of test samples to use", required=False, default=10)
    parser.add_argument('-ts', metavar='TRAIN_STRATEGY', type=str, help="device to use for training", required=False, default='GPU')
    args = parser.parse_args()

    model_path = os.path.join(args.p, 'tf_model')

    json_path = os.path.join(args.p, 'run.json')
    print('Reading configuration...')
    config = read_json_config(json_path)

    if args.d is not None:
        config.dataset.path = args.d
    # reduce batch size to 1
    config.dataset.val_test_batch_size = 1

    train_strategy = initialize_train_strategy(args.ts, True)

    # Load and prepare the dataset
    print('Preparing datasets...')
    dataset_generator = dataset_generator_factory(config.dataset, config.others)
    test_ds, classes_count, image_shape, test_batches = dataset_generator.generate_test_dataset()
    print('Datasets generated successfully')

    # Generate the model
    with train_strategy.scope():
        model = models.load_model(model_path)   # type: Model
    print('Model loaded successfully from TF model files')

    save_folder = os.path.join(args.p, 'test-predictions')
    create_empty_folder(save_folder)

    # limit the dataset to the indicated number of samples
    test_samples = test_ds.take(args.n)

    # plot some predictions
    for i, (x, y) in enumerate(test_samples):
        # use the second output ([1])
        mask = model.predict(x)[1]

        # Convert the mask output to an image with pixel values in the range [0, 255]
        mask_img = np.argmax(mask, axis=-1)[0]

        # Display the input image and the mask output image side by side
        fig, axs = plt.subplots(1, 3)   # type: plt.Figure, plt.Axes

        # set the figure dimension based on the image size
        _, h, w, _ = [dim / 100 for dim in x.get_shape().as_list()]  # assuming the usual NHWC format
        fig.set_size_inches(w * 3 + 2, h + 0.6)

        # cast to float32, since fp16 is not supported by matplotlib and model could be in mixed precision
        sample = np.asarray(x[0], dtype=np.float32)
        true_labels = np.asarray(np.squeeze(y[0]), dtype=int)
        pred_labels = np.asarray(mask_img, dtype=int)

        # convert the class labels to a common color palette
        # these two arrays contain the label (as 3 elements array) and the respective RGB value to map
        # class_vals = np.asarray(list(map(lambda k: [k] * 3, cmap_dict_rgb.keys())), dtype=int)
        class_vals = np.asarray([[k] * 3 for k in cmap_dict_rgb.keys()], dtype=int)
        rgb_values = np.asarray(list(cmap_dict_rgb.values()), dtype=int)

        # use 3 channels
        true_labels = np.stack([true_labels] * 3, axis=-1)
        pred_labels = np.stack([pred_labels] * 3, axis=-1)

        # convert the class values to the colors
        for class_v, rgb in zip(class_vals, rgb_values):
            true_labels = np.where(true_labels == class_v, rgb, true_labels)
            pred_labels = np.where(pred_labels == class_v, rgb, pred_labels)

        # convert from float to [0, 255] domain
        sample_rgb = sample * 255   # type: np.ndarray
        # convert numpy arrays to pillow images (must cast to uint8, otherwise they do not work!)
        sample_img = Image.fromarray(sample_rgb.astype(np.uint8), mode='RGB')
        true_labels_img = Image.fromarray(true_labels.astype(np.uint8), mode='RGB')
        pred_labels_img = Image.fromarray(pred_labels.astype(np.uint8), mode='RGB')
        alpha_mask = Image.new('L', sample_img.size, 80)

        true_labels_img = Image.composite(sample_img, true_labels_img, alpha_mask)
        pred_labels_img = Image.composite(sample_img, pred_labels_img, alpha_mask)

        axs[0].imshow(sample_img)
        axs[1].imshow(true_labels_img)
        axs[2].imshow(pred_labels_img)

        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, f'test_{i}.png'))


if __name__ == '__main__':
    main()
