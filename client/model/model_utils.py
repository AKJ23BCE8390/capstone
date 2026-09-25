import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    """Calculates total trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def verify_model_input_shape(model: nn.Module, input_shape: tuple, device: torch.device) -> bool:
    """Verifies forward pass correctness for a target input shape (e.g., (1, 1, 64, 64))."""
    model.eval()
    try:
        dummy_input = torch.randn(*input_shape).to(device)
        with torch.no_grad():
            output = model(dummy_input)
        return output.ndim == 2 and output.shape[0] == input_shape[0]
    except Exception as e:
        print(f"Model shape verification failed: {e}")
        return False