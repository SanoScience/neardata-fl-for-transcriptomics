import argparse
import socket

import flwr as fl

from core.utils.cifar_10 import weighted_average

parser = argparse.ArgumentParser("federated_client")
parser.add_argument("--server-ip", dest="server_ip",
                    help="ip of the server", default="localhost")
args = parser.parse_args()

server_hostname = socket.gethostname()
print("Server hostname: ", server_hostname)
print("Server address: ", socket.gethostbyname(server_hostname))

strategy = fl.server.strategy.FedAvg(
    fraction_evaluate=1.0,
    min_available_clients=2,
    evaluate_metrics_aggregation_fn=weighted_average,
)

fl.server.start_server(
    server_address=f"{args.server_ip}:8081", config=fl.server.ServerConfig(num_rounds=3), strategy=strategy,
)
