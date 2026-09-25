import os
from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from client.evaluate import evaluate_model


def load_centralized_data(data_dir: str) -> Tuple[TensorDataset, TensorDataset]:
    """Loads centralized baseline train and validation tensors."""
    train_path = os.path.join(data_dir, "train.pt")
    val_path = os.path.join(data_dir, "val.pt")

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(
            f"Centralized dataset tensors missing in {data_dir}. Run download/partition scripts first."
        )

    train_data = torch.load(train_path)
    val_data = torch.load(val_path)

    return (
        TensorDataset(train_data["images"], train_data["labels"]),
        TensorDataset(val_data["images"], val_data["labels"])
    )


def train_centralized_model(model: nn.Module, config: dict, device: torch.device) -> dict:
    """Executes full centralized baseline training loop."""
    data_dir = config["dataset"]["centralized_dir"]
    train_ds, val_ds = load_centralized_data(data_dir)

    train_loader = DataLoader(
        train_ds, batch_size=config["training"]["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["training"]["batch_size"], shuffle=False
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )

    model.to(device)
    best_acc = 0.0
    best_stats = {}

    for epoch in range(1, config["training"]["epochs"] + 1):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            if labels.ndim > 1 and labels.shape[1] == 1:
                labels = labels.squeeze(1).long()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_ds)
        val_loss, val_acc, _, _ = evaluate_model(model, val_loader, criterion, device)

        print(
            f"Epoch [{epoch:02d}/{config['training']['epochs']:02d}] | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_stats = {
                "epoch": epoch,
                "best_accuracy": best_acc,
                "train_loss": train_loss,
                "val_loss": val_loss
            }
            # Save checkpoint state dict
            os.makedirs(config["paths"]["checkpoint_dir"], exist_ok=True)
            ckpt_path = os.path.join(
                config["paths"]["checkpoint_dir"], config["paths"]["model_save_name"]
            )
            torch.save(model.state_dict(), ckpt_path)

    return best_stats