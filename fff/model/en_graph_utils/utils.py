import torch


def remove_mean(x):
    mean = torch.mean(x, dim=-2, keepdim=True)
    x = x - mean
    return x


def remove_mean_with_mask(x, node_mask):
    try:
        masked_max_abs_value = (x * (1 - node_mask)).abs().sum().item()
        assert masked_max_abs_value < 1e-5, f'Error {masked_max_abs_value} too high'
    except RuntimeError:
        pass
    N = node_mask.sum(-2, keepdims=True)

    mean = torch.sum(x, dim=-2, keepdim=True) / N
    x = x - mean * node_mask
    return x

