import argparse
import os.path
import shutil
from os import DirEntry

import pandas as pd
from tqdm import tqdm

from utils.func_utils import create_empty_folder


def generate_train_split(folder_path: str):
    new_folder = os.path.join(folder_path, 'keras_training')
    original_folder = os.path.join(folder_path, 'train')
    create_empty_folder(new_folder)

    print('Converting train folder structure...')
    class_subfolders = [f for f in os.scandir(original_folder) if f.is_dir()]  # type: list[DirEntry]
    for sf in tqdm(class_subfolders):
        new_class_folder = os.path.join(new_folder, sf.name)
        create_empty_folder(new_class_folder)

        img_folder = os.path.join(sf.path, 'images')
        for img in os.scandir(img_folder):
            shutil.copy(img.path, os.path.join(new_class_folder, img.name))


def generate_test_split(folder_path: str):
    new_folder = os.path.join(folder_path, 'keras_test')
    # test is provided but has no labels, use validation as test
    original_folder = os.path.join(folder_path, 'val')
    create_empty_folder(new_folder)

    # labels are provided in txt (csv), so we need to extract them and create folders accordingly...
    labels_csv_path = os.path.join(original_folder, 'val_annotations.txt')
    annotations_df = pd.read_csv(labels_csv_path, sep='\t', names=['filename', 'class', 'b1', 'b2', 'b3', 'b4'], usecols=['filename', 'class'])

    class_names = set(annotations_df['class'].to_list())
    print('Creating test class folders...')
    for cname in tqdm(class_names):
        create_empty_folder(os.path.join(new_folder, cname))

    print('Distributing test images to class folders...')
    img_folder = os.path.join(original_folder, 'images')
    sample_iter = zip(annotations_df['filename'], annotations_df['class'])
    for fname, cname in tqdm(sample_iter):
        new_class_folder = os.path.join(new_folder, cname)
        shutil.copy(os.path.join(img_folder, fname), os.path.join(new_class_folder, fname))


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-p', metavar='PATH', type=str, help="path to dataset folder", required=True)
    args = parser.parse_args()

    generate_train_split(args.p)
    generate_test_split(args.p)


if __name__ == '__main__':
    main()
