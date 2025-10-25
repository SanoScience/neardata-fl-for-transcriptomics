from typing import List, TypedDict

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class NumExamples(TypedDict):
    trainset: int
    testset: int


class _CSVTensorDataset(Dataset):
    def __init__(self, data_path: str):
        self.data = pd.read_csv(data_path, header=None).values
        self.data = torch.tensor(self.data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample, sample


class DeepMicroDataset:
    def __init__(
        self,
        data_path: str,
        data_split_indices: List[int],
        train_test_ratio: float = 0.7,
    ) -> None:

        self.full_dataset = _CSVTensorDataset(data_path)

        n_train_samples = round(len(data_split_indices) * train_test_ratio)
        self.train_indices = data_split_indices[:n_train_samples]
        self.test_indices = data_split_indices[n_train_samples:]

    def get_trainloader(
        self, batch_size: int = 32, shuffle: bool = True, num_workers: int = 0
    ) -> DataLoader:
        train_subset = torch.utils.data.Subset(self.full_dataset, self.train_indices)
        return torch.utils.data.DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

    def get_testloader(
        self, batch_size: int = 32, shuffle: bool = True, num_workers: int = 0
    ) -> DataLoader:
        test_subset = torch.utils.data.Subset(self.full_dataset, self.test_indices)
        return DataLoader(
            test_subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

    def get_num_examples(self) -> NumExamples:
        return NumExamples(
            trainset=len(self.train_indices), testset=len(self.test_indices)
        )
