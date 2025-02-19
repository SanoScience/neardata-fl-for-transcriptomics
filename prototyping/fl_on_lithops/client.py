from typing import List, OrderedDict
import lithops
import numpy as np
import torch
import torch.nn as nn

from core.datasets.mnist import MNIST
from core.models.mnist.SimpleCNN import SimpleCNN
from core.utils.server_utils import get_device


class LithopsFlClient:
    def __init__(self, net: nn.Module, n_workers: int, client_id: str) -> None:
        super().__init__()
        self.net = net
        self.n_workers = n_workers
        self.datasets = [
            MNIST(root=DEFAULT_DATASET_ROOT, data_split_indices=indices)
            for indices in get_data_split_indices(3, 3)
        ]
        self.device = get_device()
        self.client_id = client_id
        self.fexec = lithops.FunctionExecutor()

    def train(self, epochs_num, lr, momentum, batch_size):
        dataloaders = [dataset.get_trainloader(batch_size) for dataset in self.datasets]
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.net.parameters(), lr, momentum)
        for i in range(epochs_num):
            print(f"epoch {i}")
            parameters_list = []
            # lithops
            # TODO: how to alleviate the not enough memory issue?
            fexec = lithops.FunctionExecutor(
                config={
                    "lithops": {
                        "backend": "localhost",
                        "storage": "localhost",
                        "data_limit": 1000,
                    }
                }
            )
            fexec.map(
                perform_training_step,
                self.datasets,
                extra_args=[self.net, self.device, criterion, optimizer, batch_size],
            )
            parameters_list = fexec.get_result()
            avg_parameters = average_parameters(parameters_list)
            set_parameters(self.net, avg_parameters)


def perform_training_step(dataset, net, device, criterion, optimizer, batch_size):
    loader = dataset.get_trainloader(batch_size)
    for images, labels in loader:
        images, _ = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(net(images), labels)
        loss.backward()
        optimizer.step()
    return get_parameters(net)


def get_parameters(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def average_parameters(parameters_list):
    avgd_params = []
    for i in range(len(parameters_list[0])):
        avgd_params.append(
            np.average(np.stack([params[i] for params in parameters_list]), axis=0)
        )
    return avgd_params


def set_parameters(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_data_split_indices(n_samples: int, n_splits: int) -> List:
    partition_size = n_samples // n_splits
    shuffled_indices = torch.randperm(n_samples).tolist()
    return [
        shuffled_indices[i : i + partition_size]
        for i in range(0, n_samples, partition_size)
    ]


if __name__ == "__main__":
    DEFAULT_DATASET_ROOT = "./data/raw/mnist"
    data_split_indices = get_data_split_indices(10000, 3)
    n_workers = 3
    client = LithopsFlClient(net=SimpleCNN(), n_workers=n_workers, client_id="test")
    client.train(epochs_num=10, lr=0.001, momentum=0.1, batch_size=64)
