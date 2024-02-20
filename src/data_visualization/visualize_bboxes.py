import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torchvision.io import read_image
import yaml


image = Image.open(
    "../../data/roboflow/all2/images/macropus_fuliginosus_162_jpg.rf.9c255bfcfb90f83c637ab3a7b43389ee.jpg")
image_width, image_height = image.size

# read yaml
with open("../../data/roboflow/data.yaml", "r") as f:
    yaml_data = yaml.safe_load(f)

# Read YOLO format bounding box from file
with open("../../data/roboflow/all2/labels/macropus_fuliginosus_162_jpg.rf.9c255bfcfb90f83c637ab3a7b43389ee.txt", "r") as f:
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
                             linewidth=1, edgecolor='r', facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(rect)
    ax.text(x1+5, y1+25, yaml_data['names'][label],
            color='r', fontsize=10)

plt.show()
