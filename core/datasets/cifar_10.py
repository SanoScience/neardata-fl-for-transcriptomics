from typing import TypedDict

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class NumExamples(TypedDict):
    trainset: int
    testset: int


class CIFAR10:
    def __init__(self, root: str) -> None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.trainset = torchvision.datasets.CIFAR10(
            root=root, train=True, download=True, transform=transform)
        self.testset = torchvision.datasets.CIFAR10(
            root=root, train=False, download=True, transform=transform)

    def get_trainloader(self, batch_size: int = 32, shuffle: bool = True, num_workers: int = 2) -> DataLoader:
        return torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def get_testloader(self, batch_size: int = 32, shuffle: bool = True, num_workers: int = 2) -> DataLoader:
        return DataLoader(self.testset,  batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def get_num_examples(self) -> NumExamples:
        return NumExamples(trainset=len(self.trainset), testset=len(self.testset))
