from PIL import Image
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
import torch

from faster_rcnn.utils.custom_utils import ROOT_DIR


def get_transform(train):
    """
    Define an image transformation pipeline similar to the YOLO default configuration
    set in hyp.scratch-high.yaml.

    :param: train (bool): A flag indicating whether the transformation pipeline is for training or not.

    :returns:
        torchvision.transforms.v2.Compose: A composed transformation that includes resize, crop,
        color jitter, random affine, normalization, and tensor conversion.
    """
    transforms = []

    if train:
        # Augmentations for training
        transforms.extend([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomApply(torch.nn.ModuleList([
                v2.RandomOrder([
                    v2.RandomChoice([
                        v2.ColorJitter(hue=0.015, saturation=0.7, brightness=0.4),
                        v2.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 0.9), shear=0),
                    ])
                ])
            ]), p=0.9),
            v2.RandomOrder([
                v2.Resize(size=(640, 640), interpolation=InterpolationMode.BICUBIC),
                v2.RandomCrop(size=(640, 640), pad_if_needed=True, fill=0, padding_mode="constant"),
            ]),
            v2.RandomApply(torch.nn.ModuleList([
                v2.RandomOrder([
                    v2.RandomChoice([
                        v2.ColorJitter(hue=0.015, saturation=0.7, brightness=0.4),
                        v2.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 0.9), shear=0),
                        ]),
                    ])
            ]), p=0.9),
        ])

    else:
        # Augmentations for validation/testing
        transforms.append(v2.Resize(size=(640, 640), interpolation=InterpolationMode.BICUBIC))

    # Common transformations
    transforms.extend([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return v2.Compose(transforms)


def check_transform():
    """Test the transformation function."""
    # Define a dummy image path for testing
    image_path = ROOT_DIR / 'data'/'images'/'test'/'images'/('osphranter_rufus_044_jpg.rf'
                                                             '.c63012b01570245275544cada94b23fa.jpg')

    # Open the image
    image = Image.open(image_path)

    # Display the original image
    plt.figure(figsize=(10, 10))
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis("off")
    plt.show()

    # Apply the transformation function to the image
    transform = get_transform(train=True)
    transformed_image = transform(image)

    # Convert the transformed tensor back to a PIL image for visualization
    transformed_image_pil = to_pil_image(transformed_image)

    # Display the transformed image
    plt.figure(figsize=(10, 10))
    plt.title("Transformed Image")
    plt.imshow(transformed_image_pil)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    check_transform()
