import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class Autoencoder(nn.Module):
    """
    Fully connected auto-encoder model, symmetric.
    dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
          The decoder is symmetric with encoder.
    """

    def __init__(
        self,
        dims: List[int],
        act: str = "relu",
        latent_act: bool = False,
        output_act: bool = False,
    ):
        super(Autoencoder, self).__init__()

        encoder_layers = []
        for i in range(len(dims) - 1):
            encoder_layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # Not the latent layer
                encoder_layers.append(self._get_activation(act))
            elif latent_act:
                encoder_layers.append(self._get_activation(act))

        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        reversed_dims = dims[::-1]
        for i in range(len(reversed_dims) - 1):
            decoder_layers.append(nn.Linear(reversed_dims[i], reversed_dims[i + 1]))
            if i < len(reversed_dims) - 2:  # Not the output layer
                decoder_layers.append(self._get_activation(act))
            elif output_act:
                decoder_layers.append(self._get_activation("sigmoid"))

        self.decoder = nn.Sequential(*decoder_layers)

    def _get_activation(self, act_str: str):
        if act_str == "relu":
            return nn.ReLU()
        elif act_str == "sigmoid":
            return nn.Sigmoid()
        else:
            return nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class VariationalAutoencoder(nn.Module):
    """Variational Autoencoder"""

    def __init__(self, dims: List[int], act: str = "relu", output_act: bool = False):
        super(VariationalAutoencoder, self).__init__()
        self.dims = dims

        encoder_layers = []
        for i in range(len(dims) - 2):
            encoder_layers.append(nn.Linear(dims[i], dims[i + 1]))
            encoder_layers.append(self._get_activation(act))
        self.encoder_base = nn.Sequential(*encoder_layers)

        self.fc_mu = nn.Linear(dims[-2], dims[-1])
        self.fc_logvar = nn.Linear(dims[-2], dims[-1])

        decoder_layers = []
        reversed_dims = dims[::-1]
        for i in range(len(reversed_dims) - 1):
            decoder_layers.append(nn.Linear(reversed_dims[i], reversed_dims[i + 1]))
            if i < len(reversed_dims) - 2:
                decoder_layers.append(self._get_activation(act))
            elif output_act:
                decoder_layers.append(self._get_activation("sigmoid"))
        self.decoder = nn.Sequential(*decoder_layers)

    def _get_activation(self, act_str: str):
        if act_str == "relu":
            return nn.ReLU()
        elif act_str == "sigmoid":
            return nn.Sigmoid()
        else:
            return nn.Identity()

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_base(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x.view(-1, self.dims[0]))
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Decoder
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 1, 2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.t_conv1(x))
        x = torch.sigmoid(self.t_conv2(x))
        return x
