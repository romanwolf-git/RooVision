from torchvision.transforms import v2
import torchvision.transforms as transforms

# YOLOv5 augmentation values
yolo_hsv_h = 0.015  # image HSV-Hue augmentation (fraction)
yolo_hsv_s = 0.7    # image HSV-Saturation augmentation (fraction)
yolo_hsv_v = 0.4    # image HSV-Value augmentation (fraction)
yolo_degrees = 0.0  # image rotation (+/- deg)
yolo_translate = 0.1  # image translation (+/- fraction)
yolo_scale = 0.5  # image scale (+/- gain)
yolo_shear = 0.0  # image shear (+/- deg)
yolo_perspective = 0.0  # image perspective (+/- fraction), range 0-0.001
yolo_flipud = 0.0  # image flip up-down (probability)
yolo_fliplr = 0.5  # image flip left-right (probability)
yolo_mosaic = 1.0  # image mosaic (probability)
yolo_mixup = 0.0  # image mixup (probability)
yolo_copy_paste = 0.0  # segment copy-paste (probability)

# Conversion to PyTorch v2 transform
pytorch_transform = v2.Compose([
    v2.ColorJitter(
        hue=yolo_hsv_h * 180,  # Convert fraction to degrees
        saturation=yolo_hsv_s * 255,  # Convert fraction to 8-bit scale
        value=yolo_hsv_v * 255  # Convert fraction to 8-bit scale
    ),
    v2.RandomRotation(degrees=yolo_degrees),
    v2.RandomAffine(
        translate=(yolo_translate, yolo_translate),
        scale=(1 - yolo_scale, 1 + yolo_scale),
        shear=yolo_shear
    ),
    v2.RandomPerspective(distortion_scale=yolo_perspective),
    v2.RandomVerticalFlip(p=yolo_flipud),
    v2.RandomHorizontalFlip(p=yolo_fliplr),
    v2.RandomApply([v2.RandomChoice([v2.Mosaic(p=yolo_mosaic)])], p=1.0),
    v2.RandomApply([v2.RandomChoice([v2.MixUp(p=yolo_mixup)])], p=1.0),
    v2.RandomApply([v2.RandomChoice([v2.CopyPaste(p=yolo_copy_paste)])], p=1.0)
])

# Now you can use this transform on your images
