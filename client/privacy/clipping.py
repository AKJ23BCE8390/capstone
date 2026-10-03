import torch

def clip_gradients(parameters, max_grad_norm: float):
    """
    Clips gradients of model parameters to a maximum Euclidean norm (C).
    """
    device = parameters[0].device if parameters else torch.device("cpu")
    max_grad_norm = float(max_grad_norm)
    
    # Compute global norm across all parameters
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters if p.grad is not None]), 2
    )
    
    clip_coef = max_grad_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            if p.grad is not None:
                p.grad.detach().mul_(clip_coef)
                
    return total_norm.item()
