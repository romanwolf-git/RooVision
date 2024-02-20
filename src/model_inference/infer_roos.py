import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torchvision import transforms
from torchvision.io import read_image
from torchvision.ops import box_convert
import torchvision.transforms.functional as F
from torchvision.utils import draw_bounding_boxes

from src.model_architecture.get_model import get_roboflow_model
from src.data_augmentation.custom_transforms import get_transform
from src.utils.custom_utils import ROOT_DIR, read_yaml

# set matplotlib defaults
plt.rcParams['figure.dpi'] = 100
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.pad_inches"] = 0

px = 1 / plt.rcParams['figure.dpi']  # pixel in inches


def get_img_path_out(img_path):
    img_path_out = img_path.replace('img', 'output')
    img_path_out = '_inf.'.join(img_path_out.rsplit('.', 1))
    return img_path_out


def read_image_as_tensor(img_path):
    img = Image.open(img_path)
    convert_tensor = transforms.ToTensor()
    img = convert_tensor(img)
    return img


def get_size_and_dpi(img):
    """
    Gets the size of the input image and calculates the size of the figure in terms of dpi.
    Returns a list with fig_size and dpi

    :param img: pyTorch tensor
    :return settings_dict: dictionary of fig_size and dpi
    """
    img_height = img.size()[1]
    img_width = img.size()[2]

    dpi = plt.rcParams['figure.dpi']
    fig_size = (img_width/dpi, img_height/dpi)

    settings_dict = {
        'figsize': fig_size,
        'dpi': dpi
    }
    return settings_dict


def save_img(img, img_path_out):
    """
    Save and show a PIL image.
    :arg img: PIL image to save and display
    :arg img_path_out: path to where to save the file
    """
    size_dpi = get_size_and_dpi(img)
    fig, ax = plt.subplots(**size_dpi)

    img = img.detach()
    img = F.to_pil_image(img)

    ax.imshow(np.asarray(img))
    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.savefig(fname=img_path_out, dpi=130)
    plt.show()


def infer_save_roboflow(model, img_path, confidence, overlap):
    """
    Infers on a local image and saves it. Uses roboflow API.
    :param model: roboflow model
    :param img_path:
    :param int confidence: confidence of the model between 0 and 100
    :param int overlap: overlap of bbox between 0 and 100
    """
    img_path_out = get_img_path_out(img_path)
    model.predict(img_path, confidence=confidence, overlap=overlap).save(img_path_out)

    return img_path_out


def infer_save_torchvision(model, img_path, confidence, overlap):
    """Infers on a local image and saves it. Uses torchvision API.
    :param model: roboflow model
    :param img_path:
    :param int confidence: confidence of the model between 0 and 100
    :param int overlap: overlap of bbox between 0 and 100
    """
    pred_json = model.predict(img_path,
                              confidence=confidence,
                              overlap=overlap).json()

    data = []
    labels = []
    # iterate over predictions and append data and labels
    # bbox format on roboflow is cxcywh
    for prediction in pred_json['predictions']:
        data.append([
            prediction['x'],
            prediction['y'],
            prediction['width'],
            prediction['height']
        ])

        labels.append(prediction['class'])

    # convert bbox format from cxcywh to xyxy
    pred_bbox = box_convert(
        boxes=torch.tensor(data),
        in_fmt='cxcywh',
        out_fmt='xyxy'
    )

    img = read_image(img_path)
    drawn_boxes = draw_bounding_boxes(
        image=img,
        boxes=pred_bbox,
        labels=labels,
        colors='blue'
    )

    img_path_out = get_img_path_out(img_path)
    save_img(drawn_boxes, img_path_out=img_path_out)


def detect_roos_torch(model, img_path, img_path_out, confidence):
    """
    Detects and classifies kangaroos, wallabies, and wallaroos in an image using a provided PyTorch model.

    :param model: A PyTorch model pre-trained for object detection and classification.
    :param img_path: Path to the image to be classified.
    :param img_path_out: Path to save the output image with bounding boxes.
    :return: Image with bounding boxes drawn around detected kangaroos, wallabies, and wallaroos.
    """

    # put model in evaluation mode, get transformations and read input image
    model.eval()
    eval_transform = get_transform(train=False)
    img = read_image_as_tensor(img_path)
    yaml_dict = read_yaml()

    with torch.no_grad():
        x = eval_transform(img)
        # convert RGBA to RGB
        x = x[:3, ...]
        predictions = model([x, ])
        pred = predictions[0]

    img = (255.0 * (img - img.min()) / (img.max() - img.min())).to(torch.uint8)
    img = img[:3, ...]

    pred_labels = []
    pred_boxes = []

    # iterate over predictions and select based on confidence
    for label, score, box in zip(pred['labels'], pred['scores'], pred['boxes']):
        if score > (confidence/100):
            pred_labels.append(f"{yaml_dict['names'][label.item()]}: {score:.3f}")
            pred_boxes.append(box.long())

    pred_boxes = torch.vstack(pred_boxes)
    output_img = draw_bounding_boxes(img, pred_boxes, pred_labels, colors='blue')

    # get image size
    img_height = img.size()[1]
    img_width = img.size()[2]

    # get pixel size
    px = 1 / plt.rcParams['figure.dpi']

    # set size of figure
    fig_size = (img_width*px, img_height*px)

    fig, ax = plt.subplots(
        figsize=fig_size,
        dpi=plt.rcParams['figure.dpi']
    )

    ax.imshow(output_img.permute(1, 2, 0))
    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.savefig(
        fname=img_path_out,
        bbox_inches='tight',
        pad_inches=0,
        dpi=130
    )


if __name__ == "__main__":

    confidence = 40
    overlap = 30  # degree of intersection between predicted bounding boxes
    img_path = os.path.join(ROOT_DIR, 'app/images/test_kangaroo.jpg')

    model = get_roboflow_model()
    # infer_save_roboflow(model=model, img_path=img_path, confidence=confidence, overlap=overlap)
    infer_save_torchvision(model=model, img_path=img_path, confidence=confidence, overlap=overlap)

    # yaml_data = read_yaml()
    # roo_classes = yaml_data['names']
    #
    # # num classes = number of roo classes + 1 for background
    # model = load_pretrained_faster_rcnn_classification(num_classes=len(roo_classes) + 1)
    #
    # weights_path = os.path.join(ROOT_DIR, 'models/faster_rcnn_all2_5_epochs.pt')
    # model.load_state_dict(torch.load(weights_path))
    # print(model)
    #
    # #####
    # state_dict_path = os.path.join(ROOT_DIR, 'models/faster_rcnn_all2_5_epochs.pt')
    # model = load_faster_rcnn(num_classes=len(roo_classes) + 1, state_dict_path=state_dict_path)
    #
    # img_path = os.path.join(ROOT_DIR, 'app/img/test_kangaroo.jpg')
    # img_path_out = img_path.replace('img', 'output')
    # detect_roos_torch(model=model, img_path=img_path, img_path_out=img_path_out, confidence=10)