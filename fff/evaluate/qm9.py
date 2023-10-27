import torch
from tqdm.auto import tqdm

from fff.evaluate.utils import load_cache


def sample_metrics(model, ckpt_file, n_samples, temperature=1.0, force_update=None, n_atoms=None):
    cache_file = ckpt_file.parents[1] / f"cache_{ckpt_file.name}_{temperature}_{n_samples}_{n_atoms}.pt"
    if load_cache(ckpt_file, cache_file, force_update=force_update):
        metrics = torch.load(cache_file)
    else:
        metrics = model.train_data.compute_metrics(model, sample_count=n_samples, n_atoms=n_atoms)
        torch.save(metrics, cache_file)
    return metrics


def metrics_by_atom_count(model, ckpt_file, n_samples, force_update=None):
    count_metrics = []
    for node_count, prevalence in tqdm(model.train_data.node_counts.items()):
        count_metrics.append({
            "count": node_count,
            "prevalence": prevalence,
            **sample_metrics(model, ckpt_file, n_samples,
                             n_atoms=node_count, force_update=force_update)
        })
    return count_metrics
