import flwr
from core.datasets.cifar_10 import CIFAR10

from core.models.cifar_10.SimpleCNN import SimpleCNN
from fl.flower_tutorial.actors.simple_client import CifarClient

DEFAULT_DATASET_ROOT = "../../../data/raw/cifar_10"
net = SimpleCNN()
dataset = CIFAR10(root=DEFAULT_DATASET_ROOT)
flwr.client.start_numpy_client(
    server_address="[::]:8080", client=CifarClient(net, dataset))
