import argparse
import os
import socket
import logging

import flwr as fl
import neptune
from dotenv import load_dotenv

from core.utils.server_utils import logged_weighted_average

load_dotenv()

logging.basicConfig()
logger = logging.getLogger("fl_tutorial_server")
logger.setLevel(logging.INFO)

run = neptune.init_run(
    project="jasiek.przybyszewski/neardata-fl-for-transcriptomics",
    api_token=os.environ.get("NEPTUNE_API_TOKEN"),
    capture_hardware_metrics=False,
)

parser = argparse.ArgumentParser("federated_server")
parser.add_argument("--server-ip", dest="server_ip",
                    help="ip of the server", default="localhost")
parser.add_argument("--num-clients", dest="num_clients",
                    help="number of clients in the federation", default=2, type=int)
args = parser.parse_args()

server_hostname = socket.gethostname()
logger.info(f"Server hostname: {server_hostname}")
logger.info(f"Server address: {socket.gethostbyname(server_hostname)}")

CONFIG = {
    "epochs_num": 10,
    "lr": 0.001,
    "momentum": 0.9,
    "batch_size": 64
}

run["parameters"] = CONFIG

strategy = fl.server.strategy.FedAvg(
    fraction_evaluate=1.0,
    min_available_clients=args.num_clients,
    evaluate_metrics_aggregation_fn=logged_weighted_average(run),
    on_fit_config_fn=lambda _: CONFIG,
)

fl.server.start_server(
    server_address=f"{args.server_ip}:8081", config=fl.server.ServerConfig(num_rounds=CONFIG["epochs_num"]), strategy=strategy,
)
