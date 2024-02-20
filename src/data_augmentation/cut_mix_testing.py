import torch
import matplotlib.pyplot as plt
from torchvision.transforms import v2
from torch.utils.data import DataLoader

from src.model_training.data_loader_macropods import MacropodDataset
from roo_vision.src.utils import read_yaml

yaml_dict = read_yaml()
NUM_CLASSES = yaml_dict['nc']  # number of classes

preproc = v2.Compose([
    v2.PILToTensor(),
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),  # to float32 in [0, 1]
    # v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # typically from ImageNet
])


def collate_fn(batch):
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


cutmix = v2.CutMix(
    num_classes=NUM_CLASSES,
    # labels_getter=get_labels
                   )
mixup = v2.MixUp(
    num_classes=NUM_CLASSES,
    # labels_getter=get_labels
)
cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

# dataset = FakeData(size=1000, num_classes=NUM_CLASSES, transform=preproc)
dataset = MacropodDataset(dataset_dir='../../data/roboflow/cut_mix_testing', transforms=preproc)
dataloader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=4,
    # shuffle=True,
    num_workers=0,
    # We need a custom collation function here, because the object detection models expect a sequence of images
    # and target dictionaries. The default collation function tries to torch.stack() each element,
    # which generally fails for object detection because the number of bounding boxes varies between images in
    # the same batch.
    collate_fn=collate_fn
)

for images, targets in dataloader:
    dimensions_plot = len(images) // 2
    fig, axs = plt.subplots(nrows=dimensions_plot, ncols=2)

    for idx in range(dimensions_plot * 2):
        ax = axs[idx // 2, idx % 2] if dimensions_plot > 1 else axs[idx % 2]
        ax.imshow(images[idx].permute(1, 2, 0))
        ax.set_title(f"Image {idx + 1}")

    plt.show()
    break


