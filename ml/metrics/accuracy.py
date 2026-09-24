import torch

def calculate_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Calculates top-1 classification accuracy."""
    if y_pred.ndim > 1 and y_pred.size(1) > 1:
        preds = torch.argmax(y_pred, dim=1)
    else:
        preds = y_pred

    correct = (preds == y_true).sum().item()
    total = y_true.numel()
    return correct / total if total > 0 else 0.0