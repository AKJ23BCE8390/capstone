import torch
import numpy as np
from typing import Union
from sklearn.metrics import recall_score


def calculate_recall(
    y_true: Union[torch.Tensor, np.ndarray], 
    y_pred: Union[torch.Tensor, np.ndarray],
    average: str = "binary"
) -> float:
    """Computes recall score across ground truth and predictions."""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    return float(recall_score(y_true, y_pred, average=average, zero_division=0))