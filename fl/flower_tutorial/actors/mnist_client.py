from typing import Dict, OrderedDict, Tuple

import flwr as fl
import torch
import torch.nn as nn
from numpy import ndarray

from core.datasets.mnist import MNIST
from core.utils.server_utils import get_device


class MnistClient(fl.client.NumPyClient):
    def __init__(self, net: nn.Module, dataset: MNIST, client_id: str) -> None:
        super().__init__()
        self.net = net
        self.dataset = dataset
        self.testloader = dataset.get_testloader()
        self.num_examples = dataset.get_num_examples()
        self.device = get_device()
        self.client_id = client_id

    def get_parameters(self, config=None) -> ndarray:
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.train(config["epochs_num"], config["lr"], config["momentum"], config["batch_size"])
        return self.get_parameters(), self.num_examples["trainset"], {}

    def evaluate(self, parameters, config) -> Tuple[float, int, Dict[str, float]]:
        self.set_parameters(parameters)
        loss, accuracy = self.test(self.testloader)
        return float(loss), self.num_examples["testset"], {'client_id': self.client_id, 'accuracy': accuracy, 'loss': float(loss)}

    def train(self, epochs_num, lr, momentum, batch_size):
        dataloader = self.dataset.get_trainloader(batch_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.net.parameters(), lr, momentum)
        for _ in range(epochs_num):
            for images, labels in dataloader:
                images, _ = images.to(), labels.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.net(images), labels)
                loss.backward()
                optimizer.step()

    def test(self, dataloader) -> Tuple[float, float]:
        criterion = nn.CrossEntropyLoss()
        correct, total, loss = 0.0, 0.0, 0.0
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.net(images)
                loss += criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        return loss, accuracy
