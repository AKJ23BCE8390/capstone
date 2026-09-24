import torch

def count_parameters(model: torch.nn.Module) -> int:
    """Calculates total trainable parameters in a PyTorch module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)