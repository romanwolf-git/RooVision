from roboflow import Roboflow
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from app.config import get_secret


def load_faster_rcnn(num_classes, state_dict_path=None):
    """
    Loads a pretrained ``Faster R-CNN`` model.
    This function loads a Faster R-CNN model pretrained on COCO dataset with a MobileNetV2 backbone.

    :param str state_dict_path: path to state_dict
    :param int num_classes: number of classes in the model
    :return: torchvision.models.detection.FasterRCNN: Pretrained Faster R-CNN model.
    """
    # Load the MobileNetV2 backbone
    backbone = torchvision.models.mobilenet_v2().features
    backbone.out_channels = 1280

    # Define anchor generator
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    # Define ROI Pooler
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )

    # Create Faster R-CNN model
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )

    # Load the model state_dict if provided
    if state_dict_path:
        model.load_state_dict(torch.load(state_dict_path))

    return model


def load_roboflow_model():
    """
    Load a model from Roboflow.
    This function authenticates with the Roboflow API loads a model
    from the 'kangaroos-wallaroos-wallabies' project.

    :return: RoboflowModel: Model loaded from Roboflow.
    """
    # get secret and authenticate
    api_key = get_secret("projects/818170133534/secrets/PRIVATE_API_ROBOFLOW/versions/latest")
    rf = Roboflow(api_key=api_key)

    # Get model from roboflow project
    project = rf.workspace().project('kangaroos-wallaroos-wallabies')
    model = project.version(18).model

    return model
