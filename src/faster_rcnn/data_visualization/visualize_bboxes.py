"""Visualize bounding boxes on an image."""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import yaml

from faster_rcnn.utils.custom_utils import ROOT_DIR

# set image and label path
IMG_PATH = (ROOT_DIR / 'data' / 'images' / 'train' / 'images' / 'macropus_fuliginosus_065_jpg.rf'
                                                                '.64268a63e06d13d135c200c7fc8a1662.jpg')
LBL_PATH = (ROOT_DIR / 'data' / 'images' / 'train' / 'labels' / 'macropus_fuliginosus_065_jpg.rf'
                                                                  '.64268a63e06d13d135c200c7fc8a1662.txt')

# open image and get width and height
image = Image.open(IMG_PATH)
image_width, image_height = image.size

# read data yaml
with open((ROOT_DIR / 'data' / 'images' / 'data.yaml'), 'r') as f:
    yaml_data = yaml.safe_load(f)

# Read YOLO format bounding box from file
with open(LBL_PATH, "r") as f:
    lines = f.readlines()
    bboxes = []
    for line in lines:
        bbox = list(map(float, line.strip().split()))
        label, x_center, y_center, box_width, box_height = bbox

        # Convert YOLO format to absolute coordinates
        x1 = int((x_center - box_width / 2) * image_width)
        y1 = int((y_center - box_height / 2) * image_height)
        x2 = int((x_center + box_width / 2) * image_width)
        y2 = int((y_center + box_height / 2) * image_height)

        bboxes.append([int(label), x1, y1, x2, y2])

# Create figure and axes
fig, ax = plt.subplots()

# Display the image
ax.imshow(image)

# Create a bounding boxes with labels
for bbox in bboxes:
    label, x1, y1, x2, y2 = bbox
    rect = patches.Rectangle(xy=(x1, y1), width=x2-x1, height=y2-y1,
                             linewidth=2, edgecolor='#2e3d49', facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(rect)
    ax.text(x1+2, y1-25, yaml_data['names'][label],
            color='#2e3d49', fontsize=10, weight='bold')

plt.show()
