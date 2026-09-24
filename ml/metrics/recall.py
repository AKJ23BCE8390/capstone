import torch

def calculate_recall(y_pred: torch.Tensor, y_true: torch.Tensor, pos_label: int = 1) -> float:
    """Calculates recall (Sensitivity) for the specified positive class."""
    if y_pred.ndim > 1 and y_pred.size(1) > 1:
        preds = torch.argmax(y_pred, dim=1)
    else:
        preds = y_pred

    tp = ((preds == pos_label) & (y_true == pos_label)).sum().item()
    fn = ((preds != pos_label) & (y_true == pos_label)).sum().item()

    return tp / (tp + fn) if (tp + fn) > 0 else 0.0