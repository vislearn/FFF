from collections import Counter
from time import time

import torch
from fff.data.qm9.analyze import analyze_stability_for_molecules
from torch.utils.data import Dataset

import logging

from fff.data.qm9.data.collate import batch_stack, drop_zeros
from fff.model.en_graph_utils.utils import remove_mean_with_mask


class ProcessedDataset(Dataset):
    """
    Data structure for a pre-processed cormorant dataset.  Extends PyTorch Dataset.

    Parameters
    ----------
    data : dict
        Dictionary of arrays containing molecular properties.
    included_species : tensor of scalars, optional
        Atomic species to include in ?????.  If None, uses all species.
    num_pts : int, optional
        Desired number of points to include in the dataset.
        Default value, -1, uses all of the datapoints.
    normalize : bool, optional
        ????? IS THIS USED?
    shuffle : bool, optional
        If true, shuffle the points in the dataset.
    subtract_thermo : bool, optional
        If True, subtracts the thermochemical energy of the atoms from each molecule in GDB9.
        Does nothing for other datasets.

    """

    def __init__(self, data, included_species=None, num_pts=-1, normalize=True, shuffle=True, subtract_thermo=True,
                 include_charges=True):

        self.data = data

        if num_pts < 0:
            self.num_pts = len(data['charges'])
        else:
            if num_pts > len(data['charges']):
                logging.warning('Desired number of points ({}) is greater than the number of data points ({}) available in the dataset!'.format(num_pts, len(data['charges'])))
                self.num_pts = len(data['charges'])
            else:
                self.num_pts = num_pts

        # If included species is not specified
        if included_species is None:
            included_species = torch.unique(self.data['charges'], sorted=True)
            if included_species[0] == 0:
                included_species = included_species[1:]

        if subtract_thermo:
            thermo_targets = [key.split('_')[0] for key in data.keys() if key.endswith('_thermo')]
            if len(thermo_targets) == 0:
                logging.warning('No thermochemical targets included! Try reprocessing dataset with --force-download!')
            else:
                logging.info('Removing thermochemical energy from targets {}'.format(' '.join(thermo_targets)))
            for key in thermo_targets:
                data[key] -= data[key + '_thermo'].to(data[key].dtype)

        self.included_species = included_species

        self.data['one_hot'] = self.data['charges'].unsqueeze(-1) == included_species.unsqueeze(0).unsqueeze(0)

        self.num_species = len(included_species)
        self.max_charge = max(included_species)
        self.include_charges = include_charges

        self.parameters = {'num_species': self.num_species, 'max_charge': self.max_charge}

        # Get a dictionary of statistics for all properties that are one-dimensional tensors.
        self.calc_stats()
        self._did_remove_h = False

        if shuffle:
            self.perm = torch.randperm(len(data['charges']))[:self.num_pts]
        else:
            self.perm = None

    @property
    def data_dim(self):
        return -1

    @property
    def cond_dim(self):
        return -1

    @property
    def node_counts(self):
        return Counter((self.data["charges"] > 0).sum(-1).tolist())

    def calc_stats(self):
        self.stats = {key: (val.mean(), val.std()) for key, val in self.data.items() if type(val) is torch.Tensor and val.dim() == 1 and val.is_floating_point()}

    @torch.no_grad()
    def sample_from_model(self, model, count, n_atoms=None):
        latent = model.get_latent(model.device)

        # Make sure they have the same dimension via collate
        xh, cond = self.collate_fn([
            {
                # Is batch of size 1, so take the first element
                key: value[0]
                for key, value in latent.sample((1,), n_nodes=n_atoms)[1].items()
            } for _ in range(count)
        ])

        # Decode and quantize
        molecules = {'one_hot': [], 'x': [], 'charges': []}
        for i in range(0, len(xh), model.hparams.batch_size):
            xh_batch = xh[i:i + model.hparams.batch_size]
            cond_batch = {
                key: value[i:i + model.hparams.batch_size]
                for key, value in cond.items()
            }

            sample = model.decode(xh_batch, cond_batch)
            sample = model.quantize([sample, cond_batch])

            molecules['x'].extend(sample[..., :3])
            molecules['one_hot'].extend(sample[..., 3:-1])
            molecules['charges'].extend(sample[..., -1:])
        molecules = {key: torch.stack(value) for key, value in molecules.items()}
        molecules['node_mask'] = molecules['x'].abs().sum(dim=-1) > 0
        return molecules

    def compute_metrics(self, model, sample_count=1000, n_atoms=None):
        # 1) Sample N molecules from the model
        start = time()
        molecules = self.sample_from_model(model, sample_count, n_atoms=n_atoms)
        duration = time() - start

        # 2) Compute validity, uniqueness, novelty
        metrics, rdkit_tuple = analyze_stability_for_molecules(molecules, {
            'atom_decoder': ['C', 'N', 'O', 'F'] if self._did_remove_h else ['H', 'C', 'N', 'O', 'F'],
            'with_h': not self._did_remove_h,
            'name': 'qm9',
        })
        metrics.update({
            'sample_time': duration,
        })

        if rdkit_tuple is not None:
            metrics.update({
                'Validity': rdkit_tuple[0][0],
                'Uniqueness': rdkit_tuple[0][1],
                'Novelty': rdkit_tuple[0][2],
            })

        return metrics

    def convert_units(self, units_dict):
        for key in self.data.keys():
            if key in units_dict:
                self.data[key] *= units_dict[key]

        self.calc_stats()

    def __len__(self):
        return self.num_pts

    def __getitem__(self, idx):
        if self.perm is not None:
            idx = self.perm[idx]

        return {key: val[idx] for key, val in self.data.items()}

    def did_remove_h(self):
        self._did_remove_h = True

    def collate_fn(self, batch):
        """
        Collation function that collates datapoints into the batch format for cormorant

        Parameters
        ----------
        batch : list of datapoints
            The data to be collated.

        Returns
        -------
        batch : dict of Pytorch tensors
            The collated data.
        """
        batch = {prop: batch_stack([mol[prop] for mol in batch]) for prop in batch[0].keys()}

        # Take absolute value for collating in latent space
        to_keep = (batch['charges'].abs().sum(0) > 0)

        batch = {key: drop_zeros(prop, to_keep) for key, prop in batch.items()}

        # Take absolute value for collating in latent space
        atom_mask = batch['charges'].abs() > 0
        batch['node_mask'] = atom_mask.unsqueeze(2).long()

        # Obtain edges
        batch_size, n_nodes = atom_mask.size()
        edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)

        # mask diagonal
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool, device=batch["charges"].device).unsqueeze(0)
        edge_mask *= diag_mask

        # edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)
        batch['edge_mask'] = edge_mask.view(batch_size, n_nodes * n_nodes, 1)

        if self.include_charges:
            batch['charges'] = batch['charges'].unsqueeze(2)
        else:
            batch['charges'] = torch.zeros(0)

        xh = torch.cat([
            remove_mean_with_mask(batch["positions"], batch['node_mask']),
            batch['one_hot'],
            batch['charges']
        ], -1)
        return xh, batch
