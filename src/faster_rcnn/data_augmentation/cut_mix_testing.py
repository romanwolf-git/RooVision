"""CutMix and MixUp testing."""

import torch
import matplotlib.pyplot as plt
from torchvision.transforms import v2
from torch.utils.data import DataLoader

from faster_rcnn.model_training.data_loader_macropods import MacropodDataset
from faster_rcnn.utils.custom_utils import read_yaml, ROOT_DIR

# Read YAML configuration with number of classes
yaml_dict = read_yaml()
NUM_CLASSES = yaml_dict['nc']

# Define preprocessing transforms
preproc = v2.Compose([
    v2.PILToTensor(),
    v2.RandomResizedCrop(size=(640, 640), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),
])

# Define CutMix or MixUp transform
cutmix_or_mixup = v2.RandomChoice([
    v2.CutMix(num_classes=NUM_CLASSES),
    v2.MixUp(num_classes=NUM_CLASSES)
])


def collate_fn(batch):
    """Collate function to handle variable length labels and apply CutMix or MixUp."""
    image_list = [item[0] for item in batch]
    label_list = [item[1]['labels'] for item in batch]

    # Check if the number of labels is equal in the batch
    are_lengths_same = all(len(item[1]['labels']) == len(batch[0][1]['labels']) for item in batch)

    if are_lengths_same:
        images = torch.stack([item[0] for item in batch])
        labels = torch.cat([item[1]['labels'] for item in batch])

        images, labels = cutmix_or_mixup(images, labels)

        # Create a new list with modified elements
        new_batch = []
        for idx, item in enumerate(batch):
            new_item = (images[idx], {'labels': labels[idx]})
            new_batch.append(new_item)

        return tuple(zip(*new_batch))

    return tuple(zip(*batch))


# Create dataset and dataloader
dataset = MacropodDataset(dataset_dir=ROOT_DIR / 'data' / 'images' / 'valid', transforms=preproc)
dataloader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=4,
    num_workers=0,
    collate_fn=collate_fn
)

# Iterate over batches and display images
for images, targets in dataloader:
    dimensions_plot = len(images) // 2
    fig, axs = plt.subplots(nrows=dimensions_plot, ncols=2)

    for idx in range(dimensions_plot * 2):
        ax = axs[idx // 2, idx % 2] if dimensions_plot > 1 else axs[idx % 2]
        ax.imshow(images[idx].permute(1, 2, 0))
        ax.set_title(f"Image {idx + 1}")

    plt.show()
    break


