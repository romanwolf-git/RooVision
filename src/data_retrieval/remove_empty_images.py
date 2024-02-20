"""
This script checks for images without annotations and deletes them.
"""

import os
from roo_vision.src.utils import ROOT_DIR


# set the directory for the labels to be checked
LABEL_DIR = os.path.join(ROOT_DIR, 'data', 'roboflow', 'all2', 'labels')


def remove_empty_images(label_dir=LABEL_DIR):
    """
    function to remove unannotated images from the dataset
    :param label_dir: label directory
    """
    for file_name in sorted(os.listdir(label_dir)):
        file_path = os.path.join(LABEL_DIR, file_name)

        with open(file_path, 'r') as f:
            lines = f.readlines()
            if len(lines) == 0:
                os.remove(file_path.replace('labels', 'images').replace('txt', 'jpg'))
                print(f'{file_path.split("/")[-1]} removed.')
                os.remove(file_path)


if __name__ == "__main__":
    remove_empty_images()
