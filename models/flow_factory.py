# helper for creating the ST-network for each coupling layer
from functools import reduce

import numpy as np
import torch

from models.real_nvp import Coupling, Reverse, Flow, ExtendedCoupling, EncodedPastFlow, RnnFlow, RnnStateFlow, \
    ExtendedFlow, STLinear, dtype, BatchNorm

model_types = {0: "RealNVP",
               2: "tcNF-base",
               4: "tcNF-stateless",
               6: "tcNF-stateful",
               8: "tcNF-cnn",
               10: "tcNF-mlp"}

model_types_parameters = {
    "default": {"hidden_multiplier": [1, 50],
                "st_net_layers": [3, 8],
                "st_dropout": [0.1, 0.9],
                "st_funnel_factor": [1, 10],
                "coupling_layers": [3, 20],
                "seed": [30, 100]},
    "RealNVP": {},
    "tcNF-base": {"past": [1, 100]},
    "tcNF-stateless": {"past": [1, 100],
                       "encoder_layers": [1, 10],
                       "encoder_dropout": [0.1, 0.9]},
    "tcNF-stateful": {"encoder_layers": [1, 10],
                      "encoder_dropout": [0.1, 0.9]},
    "tcNF-cnn": {"past": [10, 100],
                 "encoder_layers": [1, 5],
                 "encoder_size": [3, 7],
                 "encoder_dropout": [0.1, 0.9],
                 "encoder_channel_depth": [1, 20]},
    "tcNF-mlp": {"past": [1, 100],
                 "encoder_layers": [3, 20],
                 "encoder_dropout": [0.1, 0.9],
                 "encoder_compression_factor": [1, 20]},
}


def make_net(feature_dim=2, add_dim=0, st_net_layers=3, hidden_multiplier=10, st_dropout=0.4, st_funnel_factor=1,
             code_version=3):
    # ignoring this in the parameter optimization
    io_size = feature_dim // 2

    st_layer_sizes = np.linspace(10 * hidden_multiplier,
                                 int(10 * hidden_multiplier // st_funnel_factor),
                                 st_net_layers - 1).astype(int)

    layers = [torch.nn.Linear(io_size + add_dim, 10 * hidden_multiplier),
              torch.nn.Tanh(),
              torch.nn.Dropout(st_dropout), ]

    current_size = st_layer_sizes[0]
    for next_size in st_layer_sizes[1:]:
        if next_size % 2 == 1:
            # check that next size is even to have a nice split in the STLiner mask
            next_size += 1
        if code_version > 3:
            layers.append(STLinear(current_size, next_size))
        else:
            layers.append(torch.nn.Linear(current_size, next_size))
        layers.append(torch.nn.Tanh())
        layers.append(torch.nn.Dropout(st_dropout))
        current_size = next_size

    if code_version > 3:
        layers.append(STLinear(current_size, feature_dim))
    else:
        layers.append(torch.nn.Linear(current_size, feature_dim))

    return torch.nn.Sequential(*layers)


# helper for creating the identity network to pass past information directly to the next steps
def identity():
    return torch.nn.Sequential(
        torch.nn.Identity()
    )


class Permute(torch.nn.Module):

    def __init__(self, pattern, *args, **kwargs):
        super(Permute, self).__init__(*args, **kwargs)
        self.pattern = pattern

    def forward(self, x):
        x = x.permute(self.pattern)
        return x


# helper for creating the CNN-encoder network for the past information encoding
def cnn(channels=2, encoder_layers=3, encoder_size=3, encoder_dropout=0.4, encoder_channel_depth=1):
    layers = [Permute((0, 2, 1))]
    if encoder_size % 2 == 0:
        encoder_size += 1

    padding = encoder_size // 2

    out_channels = np.linspace(channels, encoder_channel_depth, encoder_layers).astype(int)

    prev_out_channels = channels
    for i in range(encoder_layers):
        layers.append(torch.nn.Conv1d(
            in_channels=prev_out_channels,
            out_channels=int(out_channels[i]),
            kernel_size=encoder_size,
            stride=1,
            padding=padding,

        ))
        layers.append(torch.nn.GELU())
        layers.append(torch.nn.Dropout1d(encoder_dropout))
        prev_out_channels = int(out_channels[i])

    layers = layers[:-2]

    return torch.nn.Sequential(*layers)


def mlp(add_dims=(20, 8), encoder_layers=3, encoder_dropout=0.4, encoder_compression_factor=1):
    full_in = reduce(lambda x_, y_: x_ * y_, add_dims)
    add = full_in // encoder_compression_factor
    comp = np.linspace(full_in, add, encoder_layers + 1).astype(int)

    layers = [torch.nn.Flatten()]
    for i in range(encoder_layers):
        layers.append(torch.nn.Linear(comp[i], comp[i + 1]))
        layers.append(torch.nn.GELU())
        layers.append(torch.nn.Dropout(encoder_dropout))
    layers = layers[:-2]

    return torch.nn.Sequential(*layers), add


# helper for creating the RNN-encoder network for the past information
def rnn(add_dim=0, encoder_layers=3, encoder_dropout=0.4):
    # TODO improve this model
    # ignoring this in the parameter optimization
    return torch.nn.Sequential(
        torch.nn.LSTM(
            input_size=add_dim,
            hidden_size=add_dim,
            num_layers=encoder_layers,
            batch_first=True,
            dropout=encoder_dropout,
            bidirectional=False,
        )
    )


def build_extended_coupling_chain(latent_dim, add_dim, coupling_layers, code_version, st_net_layers, hidden_multiplier,
                                  st_dropout, st_funnel_factor):
    couplings = [
        [ExtendedCoupling(make_net(latent_dim, add_dim, st_net_layers, hidden_multiplier,
                                   st_dropout, st_funnel_factor, code_version), code_version),
         Reverse(),
         BatchNorm(latent_dim) if code_version > 4 else None,
         ]
        for _ in range(coupling_layers)
    ]
    couplings = [layer for layers in couplings for layer in layers if layer is not None]
    couplings.append(ExtendedCoupling(make_net(latent_dim, add_dim, st_net_layers, hidden_multiplier,
                                               st_dropout, st_funnel_factor, code_version), code_version))
    return couplings


def build_normal_coupling_chain(latent_dim, coupling_layers, code_version, st_net_layers, hidden_multiplier,
                                st_dropout, st_funnel_factor):
    couplings = [
        [Coupling(make_net(latent_dim, st_net_layers=st_net_layers,
                           hidden_multiplier=hidden_multiplier,
                           st_dropout=st_dropout,
                           st_funnel_factor=st_funnel_factor,
                           code_version=code_version), code_version),
         Reverse(),
         BatchNorm(latent_dim) if code_version > 4 else None,
         ]
        for _ in range(coupling_layers)
    ]
    couplings = [layer for layers in couplings for layer in layers if layer is not None]
    couplings.append(Coupling(make_net(latent_dim, st_net_layers=st_net_layers,
                                       hidden_multiplier=hidden_multiplier,
                                       st_dropout=st_dropout,
                                       st_funnel_factor=st_funnel_factor,
                                       code_version=code_version), code_version))
    return couplings


def flow_factory(device, code_version, model_type, input_shape, hist_shape):
    # legacy code - can be dropped
    if model_type in model_types:
        model_type = model_types[model_type]
    latent_dim = input_shape[1]
    past_dims = hist_shape[1:]

    if model_type == "RealNVP":
        # vanilla flow (Real-NVP)
        def _make_real_nvp(coupling_layers=4, st_net_layers=3, hidden_multiplier=10,
                           st_dropout=0.4, st_funnel_factor=1):

            couplings = build_normal_coupling_chain(latent_dim, coupling_layers, code_version,
                                                    st_net_layers, hidden_multiplier,
                                                    st_dropout, st_funnel_factor)

            flow = Flow(latent_dim, couplings, code_version, device).to(device)
            return flow

        return _make_real_nvp

    elif model_type == "tcNF-base":
        # tcNF-base - passthrough past flow
        def _make_tcnf_base(past_fixed=False, past=3, coupling_layers=4, st_net_layers=3, hidden_multiplier=10,
                            st_dropout=0.4, st_funnel_factor=1):
            add_dim = reduce(lambda x_, y_: x_ * y_, past_dims)

            couplings = build_extended_coupling_chain(latent_dim, add_dim, coupling_layers, code_version,
                                                      st_net_layers, hidden_multiplier,
                                                      st_dropout, st_funnel_factor)

            flow = EncodedPastFlow(latent_dim, past, identity(), couplings, code_version, device).to(device)
            return flow

        return _make_tcnf_base

    elif model_type == "tcNF-cnn":
        # tcNF-cnn - CNN encoder flow
        def _make_tcnf_cnn(past_fixed=False, past=3, coupling_layers=4,
                           st_net_layers=3, hidden_multiplier=10, st_dropout=0.4, st_funnel_factor=1,
                           encoder_layers=3, encoder_size=3, encoder_dropout=0.4, encoder_channel_depth=1):

            channels = past_dims[1]
            cnn_encoder = cnn(channels, encoder_layers, encoder_size, encoder_dropout, encoder_channel_depth)
            # test the encoder to check if the network works and to get the final output dimensions
            # for the flow conditioning
            test_tensor = torch.full((1, *past_dims), 0, dtype=dtype)
            output_test = cnn_encoder.forward(test_tensor)
            encoder_output_shape = output_test.shape[1:]

            add_dim = reduce(lambda x_, y_: x_ * y_, encoder_output_shape)

            couplings = build_extended_coupling_chain(latent_dim, add_dim, coupling_layers, code_version, st_net_layers,
                                                      hidden_multiplier, st_dropout, st_funnel_factor)

            flow = EncodedPastFlow(latent_dim, past,
                                   cnn_encoder,
                                   couplings, code_version, device).to(device)
            return flow

        return _make_tcnf_cnn

    elif model_type == "tcNF-mlp":
        # tcNF-mlp - MLP encoder flow
        def _make_tcnf_mlp(past_fixed=False, past=3, coupling_layers=4,
                           st_net_layers=3, hidden_multiplier=10, st_dropout=0.4, st_funnel_factor=1,
                           encoder_layers=3, encoder_dropout=0.4, encoder_compression_factor=1):

            if past_fixed:
                _past_dims_ = past_dims
            else:
                _past_dims_ = (past, past_dims[-1])

            mlp_encoder, add_dim = mlp(_past_dims_, encoder_layers, encoder_dropout, encoder_compression_factor)

            couplings = build_extended_coupling_chain(latent_dim, add_dim, coupling_layers, code_version,
                                                      st_net_layers, hidden_multiplier,
                                                      st_dropout, st_funnel_factor)

            flow = EncodedPastFlow(latent_dim, past,
                                   mlp_encoder,
                                   couplings, code_version, device).to(device)
            return flow

        return _make_tcnf_mlp

    elif model_type == "tcNF-stateless":
        # tcNF-stateless - stateless LSTM flow
        def _make_tcnf_stateless(past_fixed=False, past=3, coupling_layers=4, st_net_layers=3, hidden_multiplier=10,
                                 st_dropout=0.4, st_funnel_factor=1, encoder_layers=3):
            add_dim = hist_shape[-1]

            couplings = build_extended_coupling_chain(latent_dim, add_dim, coupling_layers, code_version,
                                                      st_net_layers, hidden_multiplier,
                                                      st_dropout, st_funnel_factor)

            flow = RnnFlow(latent_dim, past,
                           rnn(add_dim, encoder_layers),
                           couplings, code_version, device).to(device)
            return flow

        return _make_tcnf_stateless

    elif model_type == "tcNF-stateful":
        # tcNF-stateful - stateful LSTM flow
        def _make_tcnf_stateful(past_fixed=False, past=3, coupling_layers=4, st_net_layers=3, hidden_multiplier=10,
                                st_dropout=0.4, st_funnel_factor=1,
                                encoder_layers=3):
            add_dim = hist_shape[-1]

            couplings = build_extended_coupling_chain(latent_dim, add_dim, coupling_layers, code_version,
                                                      st_net_layers, hidden_multiplier,
                                                      st_dropout, st_funnel_factor)

            flow = RnnStateFlow(latent_dim, past, encoder_layers, add_dim,
                                rnn(add_dim, encoder_layers),
                                couplings, code_version, device).to(device)
            return flow

        return _make_tcnf_stateful

    else:
        raise NotImplemented()
