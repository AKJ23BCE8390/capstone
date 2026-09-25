import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from client.evaluate import evaluate_model


def run_baseline_evaluation(model: nn.Module, config: dict, device: torch.device) -> dict:
    """Evaluates the saved baseline model against centralized test tensors."""
    test_path = os.path.join(config["dataset"]["centralized_dir"], "test.pt")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test tensor missing: {test_path}")

    test_data = torch.load(test_path)
    test_ds = TensorDataset(test_data["images"], test_data["labels"])
    test_loader = DataLoader(
        test_ds, batch_size=config["training"]["batch_size"], shuffle=False
    )

    ckpt_path = os.path.join(
        config["paths"]["checkpoint_dir"], config["paths"]["model_save_name"]
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, y_true, y_pred = evaluate_model(model, test_loader, criterion, device)

    print(f"\n--- Baseline Centralized Test Evaluation ---")
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc*100:.2f}%")

    return {
        "test_loss": test_loss,
        "test_accuracy": test_acc
    }