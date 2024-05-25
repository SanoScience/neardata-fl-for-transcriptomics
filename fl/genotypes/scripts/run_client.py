import argparse
import socket
import secrets
import logging

import pandas as pd
from sklearn.model_selection import train_test_split
import flwr as fl
from fl.genotypes.actors.lr_client import LRGenotypesClient, SklearnDataset

logging.basicConfig()
logger = logging.getLogger("genotypes_client")
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser("federated_client")
parser.add_argument("--server-ip", dest="server_ip", help="ip of the server")
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


df = pd.read_csv("prototyping/variant_classification/data/genotypes/chr4/dataset.csv")
y = df["label"].to_numpy()
X = df.loc[:, df.columns != "label"].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
dataset = SklearnDataset(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    n_features=400,
    n_classes=2,
)
client = LRGenotypesClient(sklearn_dataset=dataset, client_id=args.client_id)
fl.client.start_numpy_client(server_address=f"{args.server_ip}:8081", client=client)
