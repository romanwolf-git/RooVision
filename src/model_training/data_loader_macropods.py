import os
import torch

from torchvision.io import read_image
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torch.utils.data import Dataset

from roo_vision.src.utils import ROOT_DIR


class MacropodDataset(Dataset):
    def __init__(self, dataset_dir, transforms):
        self.root = dataset_dir
        self.transforms = transforms
        # load and sort all image files and labels
        self.image_folder = os.path.join(ROOT_DIR, dataset_dir, "images")
        self.label_folder = os.path.join(ROOT_DIR, dataset_dir, "labels")
        self.images = list(sorted(os.listdir(self.image_folder)))
        self.bbox = list(sorted(os.listdir(self.label_folder)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.image_folder, self.images[idx])
        label_path = os.path.join(self.label_folder, self.bbox[idx])

        # read image and get its width and height
        image = read_image(img_path)
        image_width = F.get_size(image)[0]
        image_height = F.get_size(image)[1]

        # open roboflow yolov8 labels

        with open(label_path, 'rb') as f:
            lines = f.readlines()

            boxes_list = []
            labels_list = []

            for line in lines:
                bbox = list(map(float, line.strip().split()))
                label, x_center, y_center, box_width, box_height = bbox

                x1 = int((x_center - box_width / 2) * image_width)
                y1 = int((y_center - box_height / 2) * image_height)
                x2 = int((x_center + box_width / 2) * image_width)
                y2 = int((y_center + box_height / 2) * image_height)

                box = torch.tensor(data=[[x1, y1, x2, y2]])
                boxes_list.append(box)

                label = torch.tensor(data=[int(label)])
                # + 1 weg?
                labels_list.append(label+1)  # necessary adjustment due to roboflow dataset

            # concatenate boxes and labels tensors along dimension 0 (rows)
            # necessary since image can have no labels
            if boxes_list:
                boxes = torch.cat(boxes_list, dim=0)
                labels = torch.cat(labels_list, dim=0)
                area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            else:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                # labels are assumed as background label which is 0 (predefined by torchvision models)
                labels = torch.zeros(1, dtype=torch.int64)
                area = torch.zeros(1, dtype=torch.float32)
        if labels is not None and boxes.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Number of boxes (shape={boxes.shape}) and number of labels (shape={labels.shape}) do not match. "
                f"Check the image {self.images[idx]} and its label {self.bbox[idx]}"
            )

        image_id = idx
        num_objs = len(boxes)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Wrap sample and targets into torchvision tv_tensors
        image = tv_tensors.Image(image)

        # compose dictionary required for the model, see
        # boxes, labels, image_id, area, iscrowd
        target = {"boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(image)),
                  "labels": labels,
                  "image_id": image_id,
                  "area": area,
                  "iscrowd": iscrowd}

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target
