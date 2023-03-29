import flwr as fl

from core.utils.cifar_10 import weighted_average

strategy = fl.server.strategy.FedAvg(
    fraction_evaluate=1.0,
    min_available_clients=2,
    evaluate_metrics_aggregation_fn=weighted_average,
)

fl.server.start_server(
    server_address="[::]:8080", config=fl.server.ServerConfig(num_rounds=3), strategy=strategy,
)
