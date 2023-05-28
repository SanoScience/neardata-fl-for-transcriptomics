import argparse
import socket
import uuid
import logging

import flwr

from core.datasets.cifar_10 import CIFAR10
from core.models.cifar_10.SimpleCNN import SimpleCNN
from fl.flower_tutorial.actors.simple_client import CifarClient

logging.basicConfig()
logger = logging.getLogger("fl_tutorial_server")
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser("federated_client")
parser.add_argument("--server-ip", dest="server_ip",
                    help="ip of the server", default="localhost")
parser.add_argument("--client-id", dest="client_id",
                    help="unique client id", default=uuid.uuid4().int & (1 << 64)-1)

args = parser.parse_args()

server_hostname = socket.gethostname()
logger.info(f"Server hostname: {server_hostname}")
logger.info(f"Server address: {socket.gethostbyname(server_hostname)}")
logger.info(f"Client ID: {args.client_id}")

DEFAULT_DATASET_ROOT = "./data/raw/cifar_10"
net = SimpleCNN()
dataset = CIFAR10(root=DEFAULT_DATASET_ROOT)
flwr.client.start_numpy_client(
    server_address=f"{args.server_ip}:8081", client=CifarClient(net, dataset, args.client_id))
