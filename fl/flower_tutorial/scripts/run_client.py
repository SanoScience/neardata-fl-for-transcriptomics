import argparse
import socket

import flwr

from core.datasets.cifar_10 import CIFAR10
from core.models.cifar_10.SimpleCNN import SimpleCNN
from fl.flower_tutorial.actors.simple_client import CifarClient


parser = argparse.ArgumentParser("federated_server")
parser.add_argument("--server-ip", dest="server_ip",
                    help="ip of the server", default="localhost")
args = parser.parse_args()

server_hostname = socket.gethostname()
print("Server hostname: ", server_hostname)
print("Server address: ", socket.gethostbyname(server_hostname))

DEFAULT_DATASET_ROOT = "./data/raw/cifar_10"
net = SimpleCNN()
dataset = CIFAR10(root=DEFAULT_DATASET_ROOT)
flwr.client.start_numpy_client(
    server_address=f"{args.server_ip}:8081", client=CifarClient(net, dataset))
