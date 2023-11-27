import torch
from torch.utils.data import TensorDataset

import math

from fff.data.utils import TrainValTest
import os


def get_sbi_dataset(name:str, root: str, num_simulations: int = 1000) -> TrainValTest:
    try:
        train_dataset = torch.load(os.path.join(root, "train.pt"))
        val_dataset = torch.load(os.path.join(root, "val.pt"))
        test_dataset = torch.load(os.path.join(root, "test.pt"))
    except FileNotFoundError:
        train_dataset, val_dataset, test_dataset = simulate_task(name, num_simulations)
    return train_dataset, val_dataset, test_dataset



def simulate_task(taskname: str, num_simulations: int = 1000, val_fraction: float = 0.1):
    import sbibm
    from sbi import inference as inference

    task = sbibm.get_task(taskname)
    simulator = task.get_simulator(max_calls=num_simulations)
    proposal = task.get_prior_dist()
    theta, x = inference.simulate_for_sbi(simulator, proposal, num_simulations=num_simulations, simulation_batch_size=1000)
    split = math.ceil(val_fraction*num_simulations)
    theta_train, theta_val, theta_test = theta[split:], theta[:split], theta[:split]    
    x_train, x_val, x_test = x[split:], x[:split], x[:split]
    return TensorDataset(theta_train, x_train), TensorDataset(theta_val, x_val), TensorDataset(theta_test, x_test)