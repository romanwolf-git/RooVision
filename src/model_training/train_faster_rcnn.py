import torch

from roo_vision.src.utils import train_one_epoch, evaluate
from src.model_architecture.get_model import load_faster_rcnn
from src.model_training.data_loader_macropods import MacropodDataset
from roo_vision.src.utils import get_transform
from roo_vision.src.utils import read_yaml


def collate_fn(batch):
    return tuple(zip(*batch))


def train_test_split(dataset_train, dataset_test, train_ratio):
    """
    function to perform the train-test split of the dataset, returns torch dataloaders
    :param class dataset_train: torch dataset to split
    :param class dataset_test: torch dataset to split
    :param float train_ratio: relative size of training data, e.g. 0.8
    :return dataloader_train, dataloader_test
    """
    # split the dataset in train and test set
    indices = torch.randperm(len(dataset_train)).tolist()

    # Calculate sizes for train and test sets
    train_size = int(train_ratio * len(dataset_train))
    test_size = len(dataset_train) - train_size

    # perform splits
    # FIX train-test sizes, seem to small
    dataset_train = torch.utils.data.Subset(dataset_train, indices[:train_size])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-test_size:])

    # create data loaders
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        # We need a custom collation function here, because the object detection models expect a sequence of images
        # and target dictionaries. The default collation function tries to torch.stack() each element,
        # which generally fails for object detection because the number of bounding boxes varies between images in
        # the same batch.
        collate_fn=collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    return data_loader_train, data_loader_test


if __name__ == '__main__':
    # train on the GPU, MPS or CPU
    # device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    device = 'cpu'
    print(f"Using {device} device")

    # use our dataset and defined transformations
    dataset_train = MacropodDataset(dataset_dir='../../data/roboflow/all2', transforms=get_transform(train=True))
    dataset_test = MacropodDataset(dataset_dir='../../data/roboflow/all2', transforms=get_transform(train=False))

    data_loader_train, data_loader_test = train_test_split(dataset_train, dataset_test, train_ratio=0.8)

    # our dataset has 8 classes, i.e. background and 7 species
    yaml_dict = read_yaml()
    num_classes = yaml_dict['nc']  # nc stands for number of classes

    # get the model using our helper function
    model = load_faster_rcnn(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1
    )

    # let's train it just for 2 epochs
    num_epochs = 5

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=int(len(data_loader_train)/10))
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    dataset_name = 'all2'
    torch.save(model.state_dict(), f"faster_rcnn_{dataset_name}_{num_epochs}_epochs.pth")
    print("That's it!")
