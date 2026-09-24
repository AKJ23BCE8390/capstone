import torch

def evaluate_model(model, dataloader, criterion, device):
    """Evaluates the model on test/validation set and computes loss/accuracy."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.squeeze().long().to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)

    total_loss = running_loss / total
    total_acc = correct / total
    return total_loss, total_acc