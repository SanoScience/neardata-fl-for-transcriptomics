import argparse
import json
import socket
import secrets
import logging

import flwr
import requests

from core.datasets.cifar_10 import CIFAR10
from core.models.cifar_10.SimpleCNN import SimpleCNN
from fl.flower_tutorial.actors.cifar_client import CifarClient

logging.basicConfig()
logger = logging.getLogger("fl_tutorial_server")
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser("federated_client")
parser.add_argument("--server-ip", dest="server_ip", help="ip of the server")
parser.add_argument(
    "--data-split-service-ip",
    dest="data_split_service_ip",
    help="ip of the data split server",
)
parser.add_argument(
    "--client-id",
    dest="client_id",
    help="unique client id",
    default=int(secrets.token_hex(4), 16),
)

args = parser.parse_args()

server_hostname = socket.gethostname()
logger.info(f"Client hostname: {server_hostname}")
logger.info(f"Client address: {socket.gethostbyname(server_hostname)}")
logger.info(f"Client ID: {args.client_id}")

data_split_server_endpoint = f"http://{args.data_split_service_ip}:8080/get_data_split"
data_split_key = "split_indices"
data_split_indices = json.loads(
    requests.get(data_split_server_endpoint).json()[data_split_key]
)

DEFAULT_DATASET_ROOT = "./data/raw/mnist"
dataset = CIFAR10(root=DEFAULT_DATASET_ROOT, data_split_indices=data_split_indices)

net = SimpleCNN()

flwr.client.start_numpy_client(
    server_address=f"{args.server_ip}:8081",
    client=CifarClient(net, dataset, args.client_id),
)
