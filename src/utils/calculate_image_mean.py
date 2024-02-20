
"""This script calculates the mean and standard deviation of a given data set."""

import os
from custom_utils import ROOT_DIR
import numpy as np
import cv2
import time

IMAGE_FOLDER = 'data/roboflow/without_empty/images'
image_names = sorted(file for file in os.listdir(IMAGE_FOLDER) if file.endswith('.jpg'))

means = np.zeros(3)
stds = np.zeros(3)

start_time = time.time()

for image_name in image_names:
    image_path = os.path.join(ROOT_DIR, IMAGE_FOLDER, image_name)
    img = cv2.imread(image_path)

    # Flatten the image for each channel
    pixels = img.reshape(-1, 3)

    # Calculate means and stds for each channel
    channel_means = np.mean(pixels, axis=0)
    channel_stds = np.std(pixels, axis=0)

    means += channel_means
    stds += channel_stds

# Calculate overall means and stds
means /= len(image_names)
stds /= len(image_names)

print(f"means: {means[0]:.2f}, {means[1]:.2f}, {means[2]:.2f}")
print(f"stds: {stds[0]:.2f}, {stds[1]:.2f}, {stds[2]:.2f}")
print(f"{time.time() - start_time:.2f}s for the calculations")
