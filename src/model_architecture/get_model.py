from roboflow import Roboflow
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from src.utils.custom_utils import get_roboflow_api_key


def load_faster_rcnn(num_classes, state_dict_path=None):
    """
    Loads a pretrained ``FasterRCNN`` model.

    :param str state_dict_path: path to state_dict
    :param int num_classes: number of classes in the model
    :return: model
    """
    # load a pre-trained model for classification and return only the features
    backbone = torchvision.models.mobilenet_v2(weights="DEFAULT").features
    # ``FasterRCNN`` needs to know the number of output channels in a backbone.
    # For mobilenet_v2, it's 1280, so we need to add it here
    # This should be equivalent to torchvision.models.mobilenet_v2(weights="DEFAULT").last_channel
    backbone.out_channels = 1280

    # let's make the RPN generate 5 x 3 anchors per spatial
    # location, with 5 different sizes and 3 different aspect
    # ratios. We have a Tuple[Tuple[int]] because each feature
    # map could potentially have different sizes and
    # aspect ratios
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    # Let's define which are the feature maps we will use to perform the region of interest
    # cropping, as well as the size of the crop after rescaling. If your backbone returns a
    # tensor, featmap_names is expected to be [0]. More generally, the backbone should return
    # an ``OrderedDict[Tensor]``, and in ``featmap_names`` you can choose which feature maps
    # to use.

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )

    # put the pieces together inside a Faster-RCNN model
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )

    # load state dict if available
    if state_dict_path:
        model.load_state_dict(torch.load(state_dict_path))

    return model


def get_roboflow_model():
    api_key = get_roboflow_api_key()
    rf = Roboflow(api_key=api_key)
    project = rf.workspace().project('kangaroos-wallaroos-wallabies')
    model = project.version(10).model
    return model
