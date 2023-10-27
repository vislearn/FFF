import torch


def sum_except_batch(x):
    return x.reshape(x.shape[0], -1).sum(dim=-1)


def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask)).abs().sum().item() < 1e-8


def compute_loss_and_nll(args, generative_model, nodes_dist, x, h, node_mask, edge_mask, context, batch_idx=None):
    bs, n_nodes, n_dims = x.size()

    edge_mask = edge_mask.view(bs, n_nodes * n_nodes)

    if args.probabilistic_model == 'diffusion':

        assert_correctly_masked(x, node_mask)

        # Here x is a position tensor, and h is a dictionary with keys
        # 'categorical' and 'integer'.
        nll = generative_model(x, h, node_mask, edge_mask, context)

        # Average over batch.
        losses = {
            "loss": nll.mean(0),
            "nll": nll.mean(0)
        }

    elif args.probabilistic_model == 'fff':
        model_out = generative_model(x, h, node_mask, edge_mask, context, batch_idx)
        losses = {
            "loss": sum(
                (weight * model_out[key]).mean(-1)
                for key, weight in args.loss_weights.items()
                if (
                        check_keys(args.loss_weights, key)
                        and (generative_model.training or key in model_out)
                )
            ),
            **{
                key: value.mean().detach()
                for key, value in model_out.items()
            }
        }

    else:
        raise ValueError(args.probabilistic_model)

    # Probability of picking this number of atoms
    N = node_mask.squeeze(2).sum(1).long()
    log_pN = nodes_dist.log_prob(N)
    losses["nll"] = losses["nll"] - log_pN.mean(-1)
    return losses


def check_keys(loss_weights, *keys):
    return any(
        (loss_key in loss_weights)
        and
        (
            torch.any(loss_weights[loss_key] > 0)
            if torch.is_tensor(loss_weights[loss_key]) else
            loss_weights[loss_key] > 0
        )
        for loss_key in keys
    )
