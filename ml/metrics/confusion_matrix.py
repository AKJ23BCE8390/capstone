import torch

def get_confusion_matrix(y_pred: torch.Tensor, y_true: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
    """Generates a confusion matrix where rows are actual labels and columns are predictions."""
    if y_pred.ndim > 1 and y_pred.size(1) > 1:
        preds = torch.argmax(y_pred, dim=1)
    else:
        preds = y_pred

    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for t, p in zip(y_true.view(-1), preds.view(-1)):
        cm[t.long(), p.long()] += 1
    return cm

def calculate_specificity(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Calculates Specificity (True Negative Rate): TN / (TN + FP)."""
    cm = get_confusion_matrix(y_pred, y_true, num_classes=2)
    tn = cm[0, 0].item()
    fp = cm[0, 1].item()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0