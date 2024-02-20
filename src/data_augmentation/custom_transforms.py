import torch
from torchvision.transforms import v2
from torchvision import tv_tensors


def get_transform(train):
    transforms = []
    if train:
        transforms.append(v2.RandomHorizontalFlip(0.5))
    transforms.append(v2.RandomCrop(size=(320, 320)))
    transforms.append(v2.ToDtype(torch.float, scale=True))
    transforms.append(v2.ToPureTensor())
    return v2.Compose(transforms)


# def get_transform(train):
#     # from detection tutorial
#     # TODO differentiate between training and testing, see e.g.
#      https://github.com/clovaai/CutMix-PyTorch/blob/master/train.py
#
#     transforms = v2.Compose(
#         [
#             v2.ToImage(),
#             # v2.RandomCrop(size=(640, 640)),  # crops an image at a random location
#             v2.Resize(size=640, antialias=True),  # test
#             v2.RandomPhotometricDistort(p=1),
#             v2.RandomZoomOut(fill={tv_tensors.Image: (123, 117, 104), "others": 0}),
#             # v2.RandomRotation(degrees=(0, 180)),  # rotates an image with random angle.
#             # v2.RandomPerspective(distortion_scale=0.6, p=1.0),  # performs random perspective transform on an image
#             v2.RandomIoUCrop(),
#             v2.RandomHorizontalFlip(p=1),
#             v2.SanitizeBoundingBoxes(),
#             v2.ToDtype(torch.float32, scale=True),
#             v2.Normalize(mean=[95.80, 113.67, 116.46], std=[45.58, 48.55, 49.61])  # calculated for the dataset
#         ]
#     )
#     return transforms

# transforms = []
# if train:
#     transforms.append(v2.RandomHorizontalFlip(0.5))
# transforms.append(v2.ToDtype(torch.float32, scale=True))
# # means: 101.57, 119.70, 121.64
# # stds: 61.80, 65.38, 66.15
# transforms.append(v2.Normalize(mean=[101.57, 119.70, 121.64], std=[61.80, 65.38, 66.15]))
# transforms.append(v2.ToPureTensor())
# return v2.Compose(transforms)


# import torchvision.transforms as transforms
#
# # YOLOv5 augmentation values
# yolo_hsv_h = 0.015  # image HSV-Hue augmentation (fraction)
# yolo_hsv_s = 0.7    # image HSV-Saturation augmentation (fraction)
# yolo_hsv_v = 0.4    # image HSV-Value augmentation (fraction)
# yolo_degrees = 0.0  # image rotation (+/- deg)
# yolo_translate = 0.1  # image translation (+/- fraction)
# yolo_scale = 0.5  # image scale (+/- gain)
# yolo_shear = 0.0  # image shear (+/- deg)
# yolo_perspective = 0.0  # image perspective (+/- fraction), range 0-0.001
# yolo_flipud = 0.0  # image flip up-down (probability)
# yolo_fliplr = 0.5  # image flip left-right (probability)
# yolo_mosaic = 1.0  # image mosaic (probability)
# yolo_mixup = 0.0  # image mixup (probability)
# yolo_copy_paste = 0.0  # segment copy-paste (probability)
#
# # Conversion to PyTorch v2 transform
# pytorch_transform = transforms.Compose([
#     transforms.ColorJitter(
#         hue=yolo_hsv_h * 180,  # Convert fraction to degrees
#         saturation=yolo_hsv_s * 255,  # Convert fraction to 8-bit scale
#         value=yolo_hsv_v * 255  # Convert fraction to 8-bit scale
#     ),
#     transforms.RandomRotation(degrees=yolo_degrees),
#     transforms.RandomAffine(
#         translate=(yolo_translate, yolo_translate),
#         scale=(1 - yolo_scale, 1 + yolo_scale),
#         shear=yolo_shear
#     ),
#     transforms.RandomPerspective(distortion_scale=yolo_perspective),
#     transforms.RandomVerticalFlip(p=yolo_flipud),
#     transforms.RandomHorizontalFlip(p=yolo_fliplr),
#     transforms.RandomApply([transforms.RandomChoice([transforms.Mosaic(p=yolo_mosaic)])], p=1.0),
#     transforms.RandomApply([transforms.RandomChoice([transforms.MixUp(p=yolo_mixup)])], p=1.0),
#     transforms.RandomApply([transforms.RandomChoice([transforms.CopyPaste(p=yolo_copy_paste)])], p=1.0)
# ])

# Now you can use this transform on your images
