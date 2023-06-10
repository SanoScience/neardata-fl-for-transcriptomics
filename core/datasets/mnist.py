from typing import List, TypedDict

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class NumExamples(TypedDict):
    trainset: int
    testset: int


class MNIST:
    def __init__(self, root: str, data_split_indices: List[int], train_test_ratio: float = 0.7) -> None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        self.full_dataset = self._get_full_dataset(root, transform)

        n_train_samples = round(len(data_split_indices)*train_test_ratio)
        self.train_indices = data_split_indices[:n_train_samples]
        self.test_indices = data_split_indices[n_train_samples:]

    def get_trainloader(self, batch_size: int = 32, shuffle: bool = True, num_workers: int = 0) -> DataLoader:
        train_subset = torch.utils.data.Subset(
            self.full_dataset, self.train_indices)
        return torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def get_testloader(self, batch_size: int = 32, shuffle: bool = True, num_workers: int = 0) -> DataLoader:
        test_subset = torch.utils.data.Subset(
            self.full_dataset, self.test_indices)
        return DataLoader(test_subset,  batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def get_num_examples(self) -> NumExamples:
        return NumExamples(trainset=len(self.train_indices), testset=len(self.test_indices))

    def _get_full_dataset(self, root: str, transform: transforms.Compose) -> torchvision.datasets.MNIST:
        trainset = torchvision.datasets.MNIST(
            root=root, train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(
            root=root, train=False, download=True, transform=transform)
        return torch.utils.data.ConcatDataset([trainset, testset])
