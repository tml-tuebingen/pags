#
# How to add a new dataset:
#
#   1. Write the loading function
#   2. Add the data set with name, loading function and number of classes in the data set dict
#

from typing import Callable, Tuple

import torch
import torch.nn.functional as F
from torchvision import transforms
import torchvision

import utils.path_config as path_config

import os


#
# Dataset Loading Functions
#
def _get_cifar_loaders(data_dir: str, dataset_fn: Callable, batch_size: int):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    train_dataset = dataset_fn(
        data_dir, train=True, download=True, transform=train_transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    test_transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = dataset_fn(data_dir, train=False, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, test_loader


def _get_cifar10_loaders(data_dir: str, batch_size: int):
    return _get_cifar_loaders(data_dir, torchvision.datasets.CIFAR10, batch_size)


def _get_cifar100_loaders(data_dir: str, batch_size: int):
    return _get_cifar_loaders(data_dir, torchvision.datasets.CIFAR100, batch_size)


def _get_mnist_loaders(data_dir: str, batch_size: int):
    dataset_fn = torchvision.datasets.MNIST
    train_transform = transforms.Compose(
        [transforms.RandomCrop(28, padding=4), transforms.ToTensor()]
    )
    train_dataset = dataset_fn(
        data_dir, train=True, download=True, transform=train_transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    test_transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = dataset_fn(data_dir, train=False, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    return [train_loader, test_loader]


def _get_tensor_dataset_loader(data_file: str, batch_size: int):
    tensor_data = torch.load(data_file)

    train_dataset = torch.utils.data.TensorDataset(
        tensor_data[0],
        tensor_data[1].to(torch.long),
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    test_dataset = torch.utils.data.TensorDataset(
        tensor_data[2], tensor_data[3].to(torch.long)
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    return [train_loader, test_loader]


def _get_word_distractor_mnist_loaders(data_dir: str, batch_size: int):
    """The mnist word distractor data set"""
    return _get_tensor_dataset_loader(
        os.path.join(data_dir, "word_distractor_mnist.pt"), batch_size
    )


def _get_simple_word_distractor_mnist_loaders(data_dir: str, batch_size: int):
    """The mnist word distractor data set"""
    return _get_tensor_dataset_loader(
        os.path.join(data_dir, "simple_word_distractor_mnist.pt"), batch_size
    )


#
# Dict to register data sets
#
# Format: Loading Function, Number of Classes
#
DATA_SET_REGISTER = {
    "cifar100": (_get_cifar100_loaders, 100),
    "cifar10": (_get_cifar10_loaders, 10),
    "mnist": (_get_mnist_loaders, 10),
    "word_distractor_mnist": (_get_word_distractor_mnist_loaders, 10),
    "simple_word_distractor_mnist": (_get_simple_word_distractor_mnist_loaders, 10),
}


#
# These functions use the info from the dict
#
def get_dataset_names():
    return DATA_SET_REGISTER.keys()


def get_num_classes(dataset_name: str) -> int:
    if dataset_name in DATA_SET_REGISTER:
        loader_fn, num_classes = DATA_SET_REGISTER[dataset_name]
    else:
        raise ValueError(f"Do not know about dataset = {dataset_name}")
    return num_classes


def get_dataloaders(
    dataset_name: str, batch_size: int
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    paths = path_config.get_paths()
    data_dir = paths[dataset_name]

    if dataset_name in DATA_SET_REGISTER:
        loader_fn, num_classes = DATA_SET_REGISTER[dataset_name]
        train_loader, test_loader = loader_fn(data_dir, batch_size)
    else:
        raise ValueError(f"Do not know about dataset = {dataset_name}")

    return train_loader, test_loader


#
# compute the test error
#
def test(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[float, int]:
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            out = model(data)
            assert not torch.isnan(out).any()
            output = F.log_softmax(out, 1)
            total_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[
                1
            ]  # get the index of the max log-probability
            # print(pred)
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

        average_loss = total_loss / len(dataloader.dataset)
        acc = 100.0 * correct / len(dataloader.dataset)
    return average_loss, acc
