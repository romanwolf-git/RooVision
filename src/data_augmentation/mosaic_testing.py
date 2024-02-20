from ultralytics.data.augment import Mosaic
from torchvision.transforms import v2
import torch
from src.model_training.data_loader_macropods import MacropodDataset
from src.data_augmentation.custom_transforms import get_transform


mosaic = Mosaic(
    dataset= MacropodDataset(dataset_dir='data/roboflow/all2', transforms=get_transform(train=True)),
    imgsz=640,
    p=1,
    n=9)


# TODO: Write custom Mosaic transformer for PyTorch like so
# the class requires a call method and can have an init method
# the class should then be included into torchvision.transformers.v2.compose
# the class should be tested separately beforehand

class Mosaic(object):
    """
    Mosaic augmentation.
    This class performs mosaic augmentation by combining multiple (4 or 9) images into a single mosaic image. The
    augmentation is applied to a dataset with a given probability.

    :arg
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        # TODO: needs to be adapted
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        # TODO: needs to be adapted
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}
