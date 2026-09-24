import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from client.data.dataset import load_medmnist_splits
from client.model.cnn import BaselineResNet18
from ml.centralized.train import train_one_epoch
from ml.centralized.evaluate import evaluate_model

def main():
    config_path = os.path.join("ml", "centralized", "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Centralized Training on Device: {device}")

    # Load Data
    train_ds, val_ds, test_ds = load_medmnist_splits(
        dataset_name=config['dataset']['name'],
        image_size=config['dataset']['image_size'],
        root_dir=config['dataset']['raw_dir']
    )

    batch_size = config['dataset']['batch_size']
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Initialize Model, Loss, Optimizer
    model = BaselineResNet18(num_classes=config['dataset']['num_classes'], pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    epochs = config['training']['epochs']
    print(f"\n--- Starting Centralized ResNet-18 Benchmark ({epochs} Epochs) ---")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")

    test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
    print("\n--- Final Centralized Benchmark Result ---")
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc*100:.2f}%")

    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = os.path.join("checkpoints", "centralized_resnet18.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved baseline checkpoint to '{checkpoint_path}'")

if __name__ == "__main__":
    main()