import torch
import numpy as np
from typing import Union


def calculate_accuracy(
    y_true: Union[torch.Tensor, np.ndarray], 
    y_pred: Union[torch.Tensor, np.ndarray]
) -> float:
    """Computes overall classification accuracy."""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    if len(y_true) == 0:
        return 0.0

    correct = np.sum(y_true == y_pred)
    return float(correct / len(y_true))