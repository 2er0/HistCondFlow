import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from loguru import logger
from torch.distributions import Normal, Independent
from torch.utils.data import Dataset

SEED = 42
TRAIN_TEST_SPLIT_SEED = SEED


def update_train_test_seed_value(seed: int):
    global TRAIN_TEST_SPLIT_SEED
    TRAIN_TEST_SPLIT_SEED = seed
    logger.warning(f"Train/Test seed value updated to {seed}")


# Set the seed for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

dtype = torch.float
DEVICE = None
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    # PyTorch does not support MPS (Mx GPU) to 100% therefore run on CPU
    DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    if os.getenv("GPU") is not None:
        DEVICE = torch.device(f"cuda:{os.getenv('GPU', 1)}")
    else:
        DEVICE = torch.device("cuda:0")
    # Seed CUDA as well
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # For multi-GPU setups
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
if DEVICE is None:
    DEVICE = torch.device("cpu")


def update_device(device):
    """
    Update the global DEVICE variable to the specified device.

    :param device: The device to set (e.g., 'cpu', 'cuda:0').
    :return: The updated device.
    """
    global DEVICE
    DEVICE = torch.device(device)
    logger.warning(f"Running with device (update): {DEVICE}")
    return DEVICE


class OpenDualDataProvider(Dataset):
    """
    Custom dataset provider for two outputs representing $x_t$ and $x_{t-k:t-1}$.
    """

    def __init__(self, data1, data2):
        """
        Initialize the dataset with two data arrays.

        :param data1: First data array.
        :param data2: Second data array.
        """
        self.data1 = data1
        self.data2 = data2

    def __getitem__(self, item):
        """
        Get the data at the specified index.

        :param item: Index of the data to retrieve.
        :return: A tuple containing data1 and data2 at the specified index.
        """
        return self.data1[item], self.data2[item]

    def __len__(self):
        """
        Get the length of the dataset.

        :return: The length of the dataset.
        """
        return self.data1.shape[0]


class Flow(nn.Module):
    """
    Default flow structure for normalizing flows with no past and passthrough past information.
    Base flow class.
    """

    def __init__(self, latent, bijections, code_version, device):
        """
        Initialize the Flow model.

        :param latent: Latent dimension.
        :param bijections: List of bijection layers.
        :param code_version: Version of the code.
        :param device: Device to run the model on.
        """
        super().__init__()
        self.device = device
        self.latent = latent
        self.normal = Independent(Normal(
            loc=torch.zeros(self.latent, device=self.device),
            scale=torch.ones(self.latent, device=self.device),
        ), 1)
        self.bijections = nn.ModuleList(bijections)
        self.code_version = code_version

    @property
    def base_dist(self):
        """
        Get the base distribution.

        :return: The base distribution.
        """
        return self.normal

    def _run_bijections(self, x, past):
        """
        Run the bijections on the input data.

        :param x: Input data.
        :param past: Past information.
        :return: Transformed data and log determinant.
        """
        log_det = torch.zeros(x.shape[0], device=self.device)
        for bijection in self.bijections:
            x, ldj = bijection(x, past)
            log_det += ldj
        return x, log_det

    def _run_forward(self, num_samples, past):
        """
        Run the forward flow transformations.

        :param num_samples: Number of samples to generate.
        :param past: Past information.
        :return: Generated samples.
        """
        z = self.base_dist.sample((num_samples,))
        z_probs = -self.base_dist.log_prob(z)
        for bijection in reversed(self.bijections):
            z, _ = bijection.inverse(z, past)
        return z, z_probs

    def log_prob(self, x, past):
        """
        Calculate the log probability of the data.

        :param x: Input data.
        :param past: Past information.
        :return: A tuple containing:
                 - full log probability
                 - distribution log probability
                 - log determinant
        """
        x, log_det = self._run_bijections(x, past)
        dist_log_prob = self.base_dist.log_prob(x)
        full_log_prob = dist_log_prob + log_det
        return full_log_prob, dist_log_prob, log_det

    def individual_log_prob(self, x, past):
        """
        Calculate the individual log probability of the data.

        :param x: Input data.
        :param past: Past information.
        :return: A tuple containing:
                 - distribution log probability
                 - log determinant
                 - transformed data
        """
        x, log_det = self._run_bijections(x, past)
        dist_log_prob = self.base_dist.base_dist.log_prob(x)
        return dist_log_prob, log_det, x

    def sample(self, num_samples, past):
        """
        Sample data from the latent space.

        :param num_samples: Number of samples to generate.
        :param past: Past information.
        :return: Generated samples.
        """
        with torch.no_grad():
            if len(past.shape) < 3 or past.shape[0] < num_samples:
                past = past[None]
                past = past.repeat(num_samples, 1, 1)
            past = past.to(dtype)
            past = past.to(self.device)

            z, z_probs = self._run_forward(num_samples, past)
            return z, z_probs

    def forward(self, x, past):
        """
        Forward processing for ONNX export.

        :param x: Input data.
        :param past: Past information.
        :return: Log probability of the data.
        """
        return self.log_prob(x, past)

    def loss(self, x, past):
        """
        Calculate the loss for the model.

        :param x: Input data.
        :param past: Past information.
        :return: A tuple containing:
                 - loss value (negative log probability)
                 - distribution log probability
                 - log determinant
        """
        full_log_prob, dist_log_prob, log_det = self.log_prob(x, past)
        return torch.mean(-full_log_prob), torch.mean(dist_log_prob), torch.mean(log_det)


class ExtendedFlow(Flow):
    """
    Flow structure for normalizing flows with no past and passthrough past information.
    """

    def __init__(self, latent, bijections, code_version, device):
        """
        Initialize the ExtendedFlow model.

        :param latent: Latent dimension.
        :param bijections: List of bijection layers.
        :param code_version: Version of the code.
        :param device: Device to run the model on.
        """
        super().__init__(latent, bijections, code_version, device)

    def _build_full_input(self, x, past):
        """
        Build the full input data for the model.

        :param x: Input data.
        :param past: Past information.
        :return: Full input data.
        """
        _past_ = past.reshape((past.shape[0], -1))
        full_x = torch.cat([x, _past_], dim=-1)
        return full_x

    def log_prob(self, x, past):
        """
        Calculate the log probability of the data.

        :param x: Input data.
        :param past: Past information.
        :return: A tuple containing:
                 - full log probability
                 - distribution log probability
                 - log determinant
        """
        full_x = self._build_full_input(x, past)
        full_log_prob, dist_log_prob, log_det = super().log_prob(full_x, past)
        return full_log_prob, dist_log_prob, log_det

    def individual_log_prob(self, x, past):
        """
        Calculate the individual log probability of the data.

        :param x: Input data.
        :param past: Past information.
        :return: A tuple containing:
                 - distribution log probability
                 - log determinant
                 - transformed data
        """
        full_x = self._build_full_input(x, past)
        dist_log_prob, log_det, x = super().individual_log_prob(full_x, past)
        return dist_log_prob, log_det, x

    def sample(self, num_samples, past):
        """
        Sample data from the latent space.

        :param num_samples: Number of samples to generate.
        :param past: Past information.
        :return: Generated samples.
        """
        # TODO fix this
        raise NotImplementedError("Sampling not implemented for ExtendedFlow")
        if len(past.shape) < 3 or past.shape[0] < num_samples:
            past = past[None]
            past = past.repeat(num_samples, 1, 1)
            past = past.to(self.device)
        past = past.to(dtype)

        z = self._run_forward(num_samples, past)
        return z


class EncodedPastFlow(Flow):
    """
    Flow structure for normalizing flows with encoded past information.
    """

    def __init__(self, latent, past, past_encoder, bijections, code_version, device):
        """
        Initialize the EncodedPastFlow model.

        :param latent: Latent dimension.
        :param past: Past information dimension.
        :param past_encoder: List of past encoder layers.
        :param bijections: List of bijection layers.
        :param code_version: Version of the code.
        :param device: Device to run the model on.
        """
        super().__init__(latent, bijections, code_version, device)
        self.past = past
        self.past_encoder = nn.ModuleList(past_encoder)

    def _run_pre_encoding(self, x, past):
        """
        Run the past encoder on the past information.

        :param x: Input data.
        :param past: Past information.
        :return: Encoded past information.
        """
        num_samples = x.shape[0]
        encoded_past = past  # torch.reshape(past, (num_samples, self.latent, self.past))

        # Run the past encoder
        for encoder in self.past_encoder:
            encoded_past = encoder(encoded_past)

        encoded_past = torch.reshape(encoded_past, (num_samples, -1))
        return encoded_past

    def log_prob(self, x, past):
        """
        Calculate the log probability of the data with encoded past information.

        :param x: Input data.
        :param past: Past information.
        :return: A tuple containing:
                 - full log probability
                 - distribution log probability
                 - log determinant
        """
        encoded_past = self._run_pre_encoding(x, past)

        # Run normalizing flow transformations
        full_log_prob, dist_log_prob, log_det = super().log_prob(x, encoded_past)
        return full_log_prob, dist_log_prob, log_det

    def individual_log_prob(self, x, past):
        """
        Calculate the individual log probability of the data with encoded past information.

        :param x: Input data.
        :param past: Past information.
        :return: A tuple containing:
                 - distribution log probability
                 - log determinant
                 - transformed data
        """
        encoded_past = self._run_pre_encoding(x, past)

        # Run normalizing flow transformations
        dist_log_prob, log_det, x = super().individual_log_prob(x, encoded_past)
        return dist_log_prob, log_det, x

    def sample(self, num_samples, past):
        """
        Sample data from the latent space with encoded past information.

        :param num_samples: Number of samples to generate.
        :param past: Past information.
        :return: Generated samples.
        """
        with torch.no_grad():
            if len(past.shape) < 3 or past.shape[0] < num_samples:
                past = past[None]
                past = past.repeat(num_samples, 1, 1)
            past = past.to(dtype)
            past = past.to(self.device)

            encoded_past = past  # torch.reshape(past, (num_samples, self.latent, self.past))

            # Run the past encoder
            for encoder in self.past_encoder:
                encoded_past = encoder(encoded_past)

            encoded_past = torch.reshape(encoded_past, (num_samples, -1))

            # Run forward flow transformations
            z, z_probs = self._run_forward(num_samples, encoded_past)

            return z, z_probs


class RnnFlow(Flow):
    """
    Flow structure for normalizing flows with stateless RNN processing.
    """

    def __init__(self, latent, past, rnn, bijections, code_version, device):
        """
        Initialize the RnnFlow model.

        :param latent: Latent dimension.
        :param past: Past information dimension.
        :param rnn: List of RNN layers.
        :param bijections: List of bijection layers.
        :param code_version: Version of the code.
        :param device: Device to run the model on.
        """
        super().__init__(latent, bijections, code_version, device)
        self.past = past
        self.rnn = nn.ModuleList(rnn)
        self.activation = nn.Tanh()

    @property
    def base_dist(self):
        """
        Get the base distribution.

        :return: The base distribution.
        """
        return self.normal

    def _run_pre_encoding(self, past):
        """
        Run the RNN encoder on the past information.

        :param past: Past information.
        :return: Encoded past information.
        """
        encoded_past = past  # torch.reshape(past, (number_samples, self.past, self.latent))

        # Run the RNN encoder
        for rnn in self.rnn:
            encoded_past, _ = rnn(encoded_past)
        encoded_past = self.activation(encoded_past)

        encoded_past = encoded_past[:, -1, :]
        return encoded_past

    def log_prob(self, x, past):
        """
        Calculate the log probability of the data with encoded past information.

        :param x: Input data.
        :param past: Past information.
        :return: A tuple containing:
                 - full log probability
                 - distribution log probability
                 - log determinant
        """
        encoded_past = self._run_pre_encoding(past)

        # Run normalizing flow transformations
        full_log_prob, dist_log_prob, log_det = super().log_prob(x, encoded_past)
        return full_log_prob, dist_log_prob, log_det

    def individual_log_prob(self, x, past):
        """
        Calculate the individual log probability of the data with encoded past information.

        :param x: Input data.
        :param past: Past information.
        :return: A tuple containing:
                 - distribution log probability
                 - log determinant
                 - transformed data
        """
        encoded_past = self._run_pre_encoding(past)

        # Run normalizing flow transformations
        dist_log_prob, log_det, x = super().individual_log_prob(x, encoded_past)
        return dist_log_prob, log_det, x

    def sample(self, num_samples, past):
        """
        Sample data from the latent space with encoded past information.

        :param num_samples: Number of samples to generate.
        :param past: Past information.
        :return: Generated samples.
        """
        # TODO check if this is correct
        if len(past.shape) < 3 or past.shape[0] < num_samples:
            past = past[None]
            past = past.repeat(num_samples, 1, 1)
            past_size = 1
        else:
            past_size = self.past
        past = past.to(dtype)
        past = past.to(self.device)

        with torch.no_grad():
            encoded_past = past  # torch.reshape(past, (num_samples, past_size, self.latent))
            for rnn in self.rnn:
                encoded_past, _ = rnn(encoded_past)

            encoded_past = self.activation(encoded_past)

            if past_size == 1:
                encoded_past = torch.squeeze(encoded_past)
            else:
                encoded_past = encoded_past[:, -1, :]

            z, z_probs = self._run_forward(num_samples, encoded_past)
        return z, z_probs


class RnnStateFlow(Flow):
    """
    Flow structure for normalizing flows with stateful RNN processing.
    """

    def __init__(self, latent, past, rnn_layers, add_dim, rnn, bijections, code_version, device):
        """
        Initialize the RnnStateFlow model.

        :param latent: Latent dimension.
        :param past: Past information dimension.
        :param rnn_layers: Number of RNN layers.
        :param add_dim: Additional dimension for RNN hidden state.
        :param rnn: List of RNN layers.
        :param bijections: List of bijection layers.
        :param code_version: Version of the code.
        :param device: Device to run the model on.
        """
        super().__init__(latent, bijections, code_version, device)
        self.rnn_size = len(rnn)
        self.rnn = nn.ModuleList(rnn)
        self.past = past
        self.rnn_layers = rnn_layers
        self.add_dim = add_dim
        self.h_c = None
        self.activation = nn.Tanh()
        self.reset_rnn_hidden()

    def reset_rnn_hidden(self):
        """
        Reset the hidden state of the RNN to a random state.

        :return: None
        """
        self.h_c = [(torch.rand((self.rnn_layers, 1, self.add_dim)).to(self.device),
                     torch.rand((self.rnn_layers, 1, self.add_dim)).to(self.device))]

    def detach(self):
        """
        Detach the hidden state of the RNN to allow for backpropagation to be more efficient after a few steps.

        :return: None
        """
        for i in range(self.rnn_size):
            self.h_c[i] = (self.h_c[i][0].detach(), self.h_c[i][1].detach())

    def _run_pre_encoding(self, past):
        """
        Run the RNN encoder on the past information.

        :param past: Past information.
        :return: Encoded past information.
        """
        encoded_past = past  # torch.reshape(past, (number_samples, self.past, self.latent))
        for i, rnn in enumerate(self.rnn):
            encoded_past, self.h_c[i] = rnn(encoded_past, self.h_c[i])

        encoded_past = self.activation(encoded_past)
        encoded_past = encoded_past[:, -1, :]
        return encoded_past

    def log_prob(self, x, past):
        """
        Normalization from the data space to the latent space (inverse processing).

        :param x: x_t input data.
        :param past: x_{t-k:t-1} past information.
        :return: A tuple containing:
                 - full log probability
                 - distribution log probability
                 - log determinant
        """
        encoded_past = self._run_pre_encoding(past)

        # Run normalizing flow transformations
        full_log_prob, dist_log_prob, log_det = super().log_prob(x, encoded_past)
        return full_log_prob, dist_log_prob, log_det

    def individual_log_prob(self, x, past):
        """
        Normalization from the data space to the latent space (inverse processing).

        :param x: x_t input data.
        :param past: x_{t-k:t-1} past information.
        :return: A tuple containing:
                 - distribution log probability
                 - log determinant
                 - transformed data
        """
        encoded_past = self._run_pre_encoding(past)

        # Run normalizing flow transformations
        dist_log_prob, log_det, x = super().individual_log_prob(x, encoded_past)
        return dist_log_prob, log_det, x

    def sample(self, num_samples, past):
        """
        Sampling from the latent space to the data space (forward processing).

        :param num_samples: Number of samples to generate.
        :param past: x_{t-k:t-1} past information.
        :return: x_t output data.
        """
        past = past.to(dtype)
        past = past.to(self.device)
        # TODO check if this is correct
        raise NotImplementedError("Sampling not implemented for RnnStateFlow, needs checking")


# Implemented Bijections
class Reverse(nn.Module):
    """
    Reverse bijection layer for normalizing flows.
    """

    def setup(self, latent):
        """
        Setup method for the Reverse layer.

        :param latent: Latent dimension.
        """
        pass

    @staticmethod
    def forward(x, past):
        """
        Forward pass for the Reverse layer.

        :param x: Input data.
        :param past: Past information.
        :return: A tuple containing:
                 - Reversed input data.
                 - Log determinant (zeros).
        """
        return x.flip(-1), x.new_zeros(x.shape[0])

    @staticmethod
    def inverse(z, past):
        """
        Inverse pass for the Reverse layer.

        :param z: Latent data.
        :param past: Past information.
        :return: A tuple containing:
                 - Reversed latent data.
                 - Log determinant (zeros).
        """
        return z.flip(-1), z.new_zeros(z.shape[0])


class Coupling(nn.Module):
    """
    Coupling bijection layer for normalizing flows.
    """

    def __init__(self, net, code_version):
        """
        Initialize the Coupling layer.

        :param net: Neural network for the coupling layer.
        :param code_version: Version of the code.
        """
        super().__init__()
        self.code_version = code_version
        self.net = net
        if self.code_version >= 3:
            self.c_alpha_d_bias = nn.Parameter(torch.rand(net[-1].out_features // 2), requires_grad=True)

    def forward(self, x, past):
        """
        Forward pass for the Coupling layer.

        :param x: Input data.
        :param past: Past information.
        :return: A tuple containing:
                 - Transformed data.
                 - Log determinant.
        """
        (z_d, x_D) = torch.chunk(x, 2, dim=-1)
        mu_and_alpha = self.net(z_d)
        mu_d, alpha_d = torch.chunk(mu_and_alpha, 2, dim=-1)

        if self.code_version == 2:
            alpha_d = torch.tanh(alpha_d)
        elif self.code_version >= 3:
            alpha_d = torch.tanh(alpha_d) * self.c_alpha_d_bias

        z_D = x_D * torch.exp(alpha_d) + mu_d
        z = torch.cat([z_d, z_D], dim=-1)

        ldj = torch.sum(alpha_d, dim=-1)
        return z, ldj

    def inverse(self, z, past):
        """
        Inverse pass for the Coupling layer.

        :param z: Latent data.
        :param past: Past information.
        :return: A tuple containing:
                 - Transformed data.
                 - Log determinant.
        """
        (x_d, z_D) = torch.chunk(z, 2, dim=-1)
        mu_and_alpha = self.net(x_d)
        mu_d, alpha_d = torch.chunk(mu_and_alpha, 2, dim=-1)

        if self.code_version == 2:
            alpha_d = torch.tanh(alpha_d)
        elif self.code_version >= 3:
            alpha_d = torch.tanh(alpha_d) * self.c_alpha_d_bias

        x_D = (z_D - mu_d) * torch.exp(-alpha_d)
        x = torch.cat([x_d, x_D], dim=-1)

        ldj = -torch.sum(-alpha_d, dim=-1)
        return x, ldj


class ExtendedCoupling(nn.Module):
    """
    Extended Coupling bijection layer for normalizing flows with past information.
    """

    def __init__(self, net, code_version):
        """
        Initialize the ExtendedCoupling layer.

        :param net: Neural network for the coupling layer.
        :param code_version: Version of the code.
        """
        super().__init__()
        self.code_version = code_version
        self.net = net
        if self.code_version >= 3:
            self.c_alpha_d_bias = nn.Parameter(torch.rand(net[-1].out_features // 2), requires_grad=True)

    def forward(self, x, past):
        """
        Forward pass for the ExtendedCoupling layer.

        :param x: Input data.
        :param past: Past information.
        :return: A tuple containing:
                 - Transformed data.
                 - Log determinant.
        """
        (z_d, x_D) = torch.chunk(x, 2, dim=-1)
        full_in = torch.cat([z_d, past], dim=-1)
        mu_and_alpha = self.net(full_in)
        mu_d, alpha_d = torch.chunk(mu_and_alpha, 2, dim=-1)

        if self.code_version == 2:
            alpha_d = torch.tanh(alpha_d)
        elif self.code_version >= 3:
            alpha_d = torch.tanh(alpha_d) * self.c_alpha_d_bias

        z_D = x_D * torch.exp(alpha_d) + mu_d
        z = torch.cat([z_d, z_D], dim=-1)

        ldj = torch.sum(alpha_d, dim=-1)
        return z, ldj

    def inverse(self, z, past):
        """
        Inverse pass for the ExtendedCoupling layer.

        :param z: Latent data.
        :param past: Past information.
        :return: A tuple containing:
                 - Transformed data.
                 - Log determinant.
        """
        (x_d, z_D) = torch.chunk(z, 2, dim=-1)
        full_in = torch.cat([x_d, past], dim=-1)
        mu_and_alpha = self.net(full_in)
        mu_d, alpha_d = torch.chunk(mu_and_alpha, 2, dim=-1)

        if self.code_version == 2:
            alpha_d = torch.tanh(alpha_d)
        elif self.code_version >= 3:
            alpha_d = torch.tanh(alpha_d) * self.c_alpha_d_bias

        x_D = (z_D - mu_d) * torch.exp(-alpha_d)
        x = torch.cat([x_d, x_D], dim=-1)

        ldj = -torch.sum(-alpha_d, dim=-1)
        return x, ldj


class STLinear(nn.Module):
    """
    Masked Linear to achieve a full ST spilt in one model
    Inspired by:
    https://github.com/zalandoresearch/pytorch-ts/blob/7860c9693d55b5c086867477cc33c89485ed0167/pts/modules/flows.py#L203
    """

    def __init__(self, in_features, out_features):
        super(STLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.in_features = self.linear.in_features
        self.out_features = self.linear.out_features

        mask = torch.full_like(self.linear.weight, 0, dtype=dtype)
        out_rows = out_features // 2
        in_cols = in_features // 2
        # set top left and the bottom right to 1 to allow data
        mask[:out_rows, :in_cols] = 1
        mask[-out_rows:, -in_cols:] = 1

        # Register the mask as a buffer so it’s not treated as a parameter
        self.register_buffer('mask', mask)

    def forward(self, x):
        # Apply the mask by element-wise multiplication
        masked_weight = self.linear.weight * self.mask
        return nn.functional.linear(x, masked_weight, self.linear.bias)


class BatchNorm(nn.Module):
    """
    BatchNorm
    Inspired by:
    https://github.com/zalandoresearch/pytorch-ts/blob/7860c9693d55b5c086867477cc33c89485ed0167/pts/modules/flows.py#L71
    """

    def __init__(self, input_size, momentum=0.9, eps=1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps

        self.log_gamma = nn.Parameter(torch.zeros(input_size))
        self.beta = nn.Parameter(torch.zeros(input_size))

        self.register_buffer("running_mean", torch.zeros(input_size))
        self.register_buffer("running_var", torch.ones(input_size))

    def forward(self, x, cond_y=None):
        if self.training:
            self.batch_mean = x.view(-1, x.shape[-1]).mean(0)
            self.batch_var = x.view(-1, x.shape[-1]).var(0, unbiased=False)

            # Update running mean and variance
            self.running_mean.mul_(self.momentum).add_(
                self.batch_mean * (1 - self.momentum)
            )
            self.running_var.mul_(self.momentum).add_(
                self.batch_var * (1 - self.momentum)
            )

            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        # Normalize the input
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        y = torch.exp(self.log_gamma) * x_hat + self.beta

        # Calculate the log-determinant of the Jacobian as a scalar
        log_abs_det_jacobian = torch.sum(self.log_gamma - 0.5 * torch.log(var + self.eps))

        return y, log_abs_det_jacobian.expand(x.shape[0])

    def inverse(self, y, cond_y=None):
        if self.training:
            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        # Invert the transformation
        x_hat = (y - self.beta) * torch.exp(-self.log_gamma)
        x = x_hat * torch.sqrt(var + self.eps) + mean

        # Calculate the inverse log-determinant of the Jacobian as a scalar
        log_abs_det_jacobian = -torch.sum(self.log_gamma - 0.5 * torch.log(var + self.eps))

        return x, log_abs_det_jacobian.expand(y.shape[0])
