import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, torch.Tensor, torch.Tensor]:
    """
    Evaluates a model performance returning Loss, Accuracy, True Labels, and Predictions.
    """
    model.to(device)
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            if labels.ndim > 1 and labels.shape[1] == 1:
                labels = labels.squeeze(1).long()

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)

            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)

            all_preds.append(preds.cpu())
            all_targets.append(labels.cpu())

    avg_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)

    y_pred = torch.cat(all_preds)
    y_true = torch.cat(all_targets)

    return avg_loss, accuracy, y_true, y_pred