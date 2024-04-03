"""Custom utility functions for the project."""
from pathlib import Path
import time
import yaml

import cv2
import numpy as np

# set project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent


def read_yaml():
    """Read data.yaml and return data"""
    yaml_path = ROOT_DIR / 'data' / 'images' / 'data.yaml'
    with open(yaml_path, 'r') as yaml_file:
        data = yaml.safe_load(yaml_file)
    return data


def calculate_image_mean(img_dir):
    """
    Calculates the mean and standard deviation of a given data set.
    :param str img_dir: path to the image directory
    """
    # Initialize arrays to store means and standard deviations
    means = np.zeros(3)
    stds = np.zeros(3)

    # Start timing
    start_time = time.time()

    # Construct full path to the image directory
    img_dir = ROOT_DIR / img_dir
    image_names = sorted(file for file in Path(img_dir).iterdir() if file.suffix == '.jpg')

    for image_name in image_names:
        # construct path and read image
        image_path = Path(ROOT_DIR) / img_dir / image_name
        img = cv2.imread(image_path)

        # Flatten the image for each channel
        pixels = img.reshape(-1, 3)

        # Calculate means and stds for each channel
        channel_means = np.mean(pixels, axis=0)
        channel_stds = np.std(pixels, axis=0)

        # Aggregate means and stds for all images
        means += channel_means
        stds += channel_stds

    # Calculate overall means and stds
    means /= len(image_names)
    stds /= len(image_names)

    print(f"means: {means[0]:.2f}, {means[1]:.2f}, {means[2]:.2f}")
    print(f"stds: {stds[0]:.2f}, {stds[1]:.2f}, {stds[2]:.2f}")
    print(f"{time.time() - start_time:.2f}s for the calculations")


if __name__ == "__main__":
    IMAGE_FOLDER = 'data/images/train/images'
    calculate_image_mean(img_dir=IMAGE_FOLDER)
