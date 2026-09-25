import torch
import numpy as np
from typing import Union, List
from sklearn.metrics import confusion_matrix


def generate_confusion_matrix(
    y_true: Union[torch.Tensor, np.ndarray], 
    y_pred: Union[torch.Tensor, np.ndarray]
) -> List[List[int]]:
    """Generates confusion matrix as a JSON-serializable 2D nested list."""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    cm = confusion_matrix(y_true, y_pred)
    return cm.tolist()