import torch

def calculate_precision(y_pred: torch.Tensor, y_true: torch.Tensor, pos_label: int = 1) -> float:
    """Calculates precision for the specified positive class (Default: Pneumonia=1)."""
    if y_pred.ndim > 1 and y_pred.size(1) > 1:
        preds = torch.argmax(y_pred, dim=1)
    else:
        preds = y_pred

    tp = ((preds == pos_label) & (y_true == pos_label)).sum().item()
    fp = ((preds == pos_label) & (y_true != pos_label)).sum().item()

    return tp / (tp + fp) if (tp + fp) > 0 else 0.0