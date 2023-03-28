from typing import OrderedDict

import flwr as fl
from numpy import ndarray
import torch
import torch.nn as nn

from core.datasets.cifar_10 import CIFAR10
from core.utils.cifar_10 import test, train


class CifarClient(fl.client.NumPyClient):
    def __init__(self, net: nn.Module, dataset: CIFAR10) -> None:
        super().__init__()
        self.net = net
        self.trainloader = dataset.get_trainloader()
        self.testloader = dataset.get_testloader()
        self.num_examples = dataset.get_num_examples()

    def get_parameters(self, config=None) -> ndarray:
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.net, self.trainloader, epochs_num=1)
        return self.get_parameters(), self.num_examples["trainset"], {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.testloader)
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}
