
import torch

from fff.model.en_graph_utils.dequantize import ArgmaxAndVariationalDequantizer
from fff.fff import FreeFormFlow, FreeFormFlowHParams


class MoleculeFreeFormFlowHParams(FreeFormFlowHParams):
    n_features: int


class MoleculeFreeFormFlow(FreeFormFlow):
    """
    A FreeFormFlow model that uses a dequantizer for the categorical and integer features.
    """
    hparams: MoleculeFreeFormFlowHParams

    def __init__(self, hparams: MoleculeFreeFormFlowHParams | dict):
        if not isinstance(hparams, MoleculeFreeFormFlowHParams):
            hparams = MoleculeFreeFormFlowHParams(**hparams)
        super().__init__(hparams)

        self.dequantizer = ArgmaxAndVariationalDequantizer(
            node_nf=self.hparams.n_features,
            device="cpu"
        )

    def _extract_h_features(self, xh, c):
        # Be sure to only use the shapes here, not the values.
        categorical_len = c["one_hot"].shape[-1]
        integer_len = c["charges"].shape[-1]
        tot_feat = xh.shape[-1]
        x = xh[:, :, :tot_feat - categorical_len - integer_len]
        assert x.shape[-1] == 3
        h = xh[:, :, tot_feat - categorical_len - integer_len:]
        categorical = h[:, :, :categorical_len]
        integer = h[:, :, categorical_len:]
        h_dict = {
            'categorical': categorical,
            'integer': integer
        }
        return x, h_dict

    def dequantize(self, batch):
        xh, c = batch

        bs, n_atoms_max, tot_feat = xh.shape
        if c is not None:
            node_mask = c["node_mask"]
        else:
            node_mask = torch.ones(bs, n_atoms_max, device=xh.device)
        if c is not None:
            edge_mask = c["edge_mask"]
        else:
            edge_mask = torch.ones(bs, n_atoms_max ** 2, device=xh.device)
        x, h_dict = self._extract_h_features(xh, c)
        out, vol_change = self.dequantizer(h_dict, node_mask, edge_mask.reshape(bs * n_atoms_max ** 2, 1), x)

        return [], torch.cat([
            x, out['categorical'], out['integer']
        ], -1), -vol_change

    def quantize(self, batch):
        xh, c = batch
        x, h_dict = self._extract_h_features(xh, c)
        h_deq = self.dequantizer.reverse(h_dict)

        return torch.cat([
            x, h_deq['categorical'], h_deq['integer']
        ], -1)
