from collections import OrderedDict

import torch.nn
from fff.model.en_graph_utils.egnn_qm9 import EGNN_dynamics_QM9
from torch import nn
from torch.nn import Module

from fff.base import ModelHParams


class ENGNNHParams(ModelHParams):
    skip_connection: bool = True

    n_dims: int
    n_features: int = 0

    mode: str = "egnn_dynamics"

    hidden_nf: int = 64
    n_layers: int = 4
    inv_sublayers: int = 2
    attention: bool = True
    norm_constant: float = 1.0


class ENGNN(Module):
    """
    This module contains two ENGNNs as encoder and decoder.
    """
    hparams: ENGNNHParams

    def __init__(self, hparams: dict | ENGNNHParams):
        if not isinstance(hparams, ENGNNHParams):
            hparams = ENGNNHParams(**hparams)

        super().__init__()
        self.hparams = hparams
        self.model = self.build_model()

        if self.hparams.latent_dim != self.hparams.data_dim and self.hparams.skip_connection:
            raise ValueError("Can only have skip connection if data_dim = latent_dim")

    def encode(self, x, c):
        return self.model_forward(self.model.encoder, x, c)

    def decode(self, z, c):
        return self.model_forward(self.model.decoder, z, c)

    def model_forward(self, model, xh, c=None):
        bs, n_atoms_max, tot_feat = xh.shape
        if isinstance(c, dict):
            node_mask = c["node_mask"]
        else:
            node_mask = torch.ones(bs, n_atoms_max, device=xh.device)
        if isinstance(c, dict):
            edge_mask = c["edge_mask"]
        else:
            edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)

            # mask diagonal
            diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool, device=xh.device).unsqueeze(0)
            edge_mask *= diag_mask
            edge_mask = edge_mask.reshape(bs, n_atoms_max, n_atoms_max, 1)

        out = model._forward(
            None, xh, node_mask, edge_mask
        )
        if self.hparams.skip_connection:
            out = xh + out
        return out

    def build_model(self) -> nn.Module:
        data_dim = self.hparams.data_dim
        if data_dim != self.hparams.latent_dim:
            raise ValueError(f"{data_dim=} != {self.hparams.latent_dim=}")

        if self.hparams.cond_dim in [-1, 0]:
            cond_dim = 0
        else:
            raise ValueError(self.hparams.cond_dim)

        # Nonlinear projection
        kwargs = dict(
            in_node_nf=self.hparams.n_features,
            context_node_nf=cond_dim,
            n_dims=self.hparams.n_dims,
            hidden_nf=self.hparams.hidden_nf,
            n_layers=self.hparams.n_layers,
            attention=self.hparams.attention,
            inv_sublayers=self.hparams.inv_sublayers,
            norm_constant=self.hparams.norm_constant,
            condition_time=False
        )
        encoder = EGNN_dynamics_QM9(**kwargs)
        decoder = EGNN_dynamics_QM9(**kwargs)

        modules = OrderedDict(
            encoder=encoder,
            decoder=decoder
        )
        return torch.nn.Sequential(modules)


ENGNNH = ENGNN
