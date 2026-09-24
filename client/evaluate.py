import torch
import torch.nn as nn
from ml.metrics.accuracy import calculate_accuracy
from ml.metrics.precision import calculate_precision
from ml.metrics.recall import calculate_recall
from ml.metrics.f1 import calculate_f1
from ml.metrics.confusion_matrix import get_confusion_matrix, calculate_specificity

def evaluate_client_model(model: nn.Module, dataloader: torch.utils.data.DataLoader, criterion: nn.Module, device: torch.device) -> dict:
    """
    Standardized client-side model evaluation function.
    
    Returns:
        Dict containing loss, accuracy, precision, recall, f1_score, specificity, and confusion_matrix.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.squeeze().long().to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            all_preds.append(outputs.cpu())
            all_targets.append(labels.cpu())

    y_pred = torch.cat(all_preds, dim=0)
    y_true = torch.cat(all_targets, dim=0)

    total_loss = running_loss / len(y_true)
    acc = calculate_accuracy(y_pred, y_true)
    prec = calculate_precision(y_pred, y_true)
    rec = calculate_recall(y_pred, y_true)
    f1 = calculate_f1(y_pred, y_true)
    spec = calculate_specificity(y_pred, y_true)
    cm = get_confusion_matrix(y_pred, y_true)

    return {
        "loss": total_loss,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "specificity": spec,
        "confusion_matrix": cm
    }