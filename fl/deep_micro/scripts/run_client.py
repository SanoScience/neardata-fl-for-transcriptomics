import argparse
import json
import logging
import secrets
import socket

import flwr
import requests

from core.datasets.deep_micro import DeepMicroDataset
from core.models.deep_micro.deep_micro_autoencoders import Autoencoder
from fl.deep_micro.actors.deep_micro_client import DeepMicroClient

logging.basicConfig()
logger = logging.getLogger("fl_deep_micro_client")
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

DEFAULT_DATASET_PATH = "core/datasets/deep_micro_data/UserDataExample.csv"
dataset = DeepMicroDataset(
    data_path=DEFAULT_DATASET_PATH, data_split_indices=data_split_indices
)

# Input dimension is 200, based on the CSV file
net = Autoencoder(dims=[200, 100, 50])

flwr.client.start_numpy_client(
    server_address=f"{args.server_ip}:8081",
    client=DeepMicroClient(net, dataset, args.client_id),
)
