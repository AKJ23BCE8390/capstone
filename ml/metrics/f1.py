import torch
import numpy as np
from typing import Union
from sklearn.metrics import f1_score


def calculate_f1(
    y_true: Union[torch.Tensor, np.ndarray], 
    y_pred: Union[torch.Tensor, np.ndarray],
    average: str = "binary"
) -> float:
    """Computes F1 score across ground truth and predictions."""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    return float(f1_score(y_true, y_pred, average=average, zero_division=0))