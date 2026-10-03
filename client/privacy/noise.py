import torch

def add_gaussian_noise(parameters, noise_multiplier: float, max_grad_norm: float):
    """
    Adds calibrated Gaussian noise to parameter gradients based on the noise multiplier and max gradient norm.
    """
    sigma = noise_multiplier * max_grad_norm
    for p in parameters:
        if p.grad is not None:
            noise = torch.normal(
                mean=0.0, 
                std=sigma, 
                size=p.grad.shape, 
                device=p.grad.device
            )
            p.grad.detach().add_(noise)
