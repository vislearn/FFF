import torch
from tqdm import tqdm

@torch.no_grad()
def c2st(model, taskname, observations=list(range(1,11))):
    from sbibm.metrics import c2st as c2st_fn
    from sbibm import get_task

    task = get_task(taskname)
    accuracy = 0.0
    for num_observation in tqdm(observations):
        reference_samples = task.get_reference_posterior_samples(num_observation=num_observation)
        observation = task.get_observation(num_observation)
        observation = observation.repeat(len(reference_samples), 1)
        z = model.get_latent("cpu").sample((len(reference_samples),))
        posterior_samples = model.decode(z, observation)
        accuracy += c2st_fn(posterior_samples, reference_samples) / len(observations)
    return accuracy
