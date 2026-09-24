import torch
from ml.metrics.precision import calculate_precision
from ml.metrics.recall import calculate_recall

def calculate_f1(y_pred: torch.Tensor, y_true: torch.Tensor, pos_label: int = 1) -> float:
    """Calculates binary F1-Score."""
    precision = calculate_precision(y_pred, y_true, pos_label)
    recall = calculate_recall(y_pred, y_true, pos_label)

    if (precision + recall) == 0:
        return 0.0
    return 2.0 * (precision * recall) / (precision + recall)