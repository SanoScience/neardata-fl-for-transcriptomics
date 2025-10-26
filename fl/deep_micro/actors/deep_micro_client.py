from typing import Dict, OrderedDict, Tuple, Union

import flwr as fl
import torch
import torch.nn as nn
from numpy import ndarray

from core.datasets.deep_micro import DeepMicroDataset
from core.models.deep_micro.deep_micro_autoencoders import (
    Autoencoder,
    ConvAutoencoder,
    VariationalAutoencoder,
)
from core.utils.server_utils import get_device


def vae_loss(recon_x, x, mu, logvar):
    # Reconstruction loss
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction="sum")
    # KL divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld


class DeepMicroClient(fl.client.NumPyClient):
    def __init__(
        self,
        net: Union[Autoencoder, VariationalAutoencoder, ConvAutoencoder],
        dataset: DeepMicroDataset,
        client_id: str,
    ) -> None:
        super().__init__()
        self.net = net
        self.dataset = dataset
        self.testloader = dataset.get_testloader()
        self.num_examples = dataset.get_num_examples()
        self.device = get_device()
        self.client_id = client_id

    def get_parameters(self, config=None) -> ndarray:
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.train(
            config["epochs_num"], config["lr"], config["momentum"], config["batch_size"]
        )
        return self.get_parameters(), self.num_examples["trainset"], {}

    def evaluate(self, parameters, config) -> Tuple[float, int, Dict[str, float]]:
        self.set_parameters(parameters)
        loss = self.test(self.testloader)
        return (
            float(loss),
            self.num_examples["testset"],
            {"client_id": self.client_id, "loss": float(loss)},
        )

    def train(self, epochs_num, lr, momentum, batch_size):
        dataloader = self.dataset.get_trainloader(batch_size)
        optimizer = torch.optim.SGD(self.net.parameters(), lr, momentum)

        for _ in range(epochs_num):
            for features, _ in dataloader:
                features = features.to(self.device)
                optimizer.zero_grad()

                if isinstance(self.net, VariationalAutoencoder):
                    recon_batch, mu, logvar = self.net(features)
                    loss = vae_loss(recon_batch, features, mu, logvar)
                else:
                    criterion = nn.MSELoss()
                    outputs = self.net(features)
                    loss = criterion(outputs, features)

                loss.backward()
                optimizer.step()

    def test(self, dataloader) -> float:
        total_loss = 0.0
        with torch.no_grad():
            for features, _ in dataloader:
                features = features.to(self.device)

                if isinstance(self.net, VariationalAutoencoder):
                    recon_batch, mu, logvar = self.net(features)
                    loss = vae_loss(recon_batch, features, mu, logvar)
                else:
                    criterion = nn.MSELoss()
                    outputs = self.net(features)
                    loss = criterion(outputs, features)
                total_loss += loss.item()

        return total_loss / len(dataloader.dataset)
