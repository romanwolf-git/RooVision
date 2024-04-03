import torch

from faster_rcnn.data_augmentation.custom_transforms import get_transform
from faster_rcnn.model_architecture.get_model import load_faster_rcnn
from faster_rcnn.model_training.data_loader_macropods import MacropodDataset
from faster_rcnn.utils.custom_utils import read_yaml, ROOT_DIR
from faster_rcnn.utils.engine import train_one_epoch, evaluate


def collate_fn(batch):
    return tuple(zip(*batch))


if __name__ == '__main__':
    # train on the GPU, MPS or CPU
    device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using {device} device")

    IMG_TRAIN_DIR = ROOT_DIR / 'data' / 'images' / 'train' / 'images'
    LABEL_TRAIN_DIR = ROOT_DIR / 'data' / 'images' / 'train' / 'labels'
    IMG_TEST_DIR = ROOT_DIR / 'data' / 'images' / 'test' / 'images'
    LABEL_TEST_DIR = ROOT_DIR / 'data' / 'images' / 'test' / 'labels'

    # load datasets
    dataset_train = MacropodDataset(img_dir=IMG_TRAIN_DIR,
                                    label_dir=LABEL_TRAIN_DIR,
                                    transforms=get_transform(train=True))
    data_loader_train = torch.utils.data.DataLoader(dataset_train,
                                                    batch_size=2,
                                                    shuffle=True,
                                                    num_workers=4,
                                                    collate_fn=collate_fn)
    dataset_test = MacropodDataset(img_dir=IMG_TEST_DIR,
                                   label_dir=LABEL_TEST_DIR,
                                   transforms=get_transform(train=False))
    data_loader_test = torch.utils.data.DataLoader(dataset_test,
                                                   batch_size=1,
                                                   shuffle=False,
                                                   num_workers=4,
                                                   collate_fn=collate_fn)

    # load data yaml
    yaml_dict = read_yaml()
    num_classes = yaml_dict['nc']  # nc = number of classes

    # get the model using our helper function
    model = load_faster_rcnn(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=0.01,  # values set as in YOLO config
        momentum=0.937,
        weight_decay=0.0005
    )

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1
    )

    num_epochs = 25

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    dataset_name = 'roboflow_v18'
    torch.save(model.state_dict(), f"faster_rcnn_{dataset_name}_{num_epochs}_epochs.pth")
    print("That's it!")
