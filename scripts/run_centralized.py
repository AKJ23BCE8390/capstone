import os
import sys
import json
import yaml
import torch

# Put root directory in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from client.model.cnn import MedicalResNet18
from ml.centralized.train import train_centralized_model
from ml.centralized.evaluate import run_baseline_evaluation


def main():
    config_path = os.path.join("ml", "centralized", "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    torch.manual_seed(config["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Centralized Baseline Model Pipeline (Device: {device}) ===")

    # Initialize Baseline Model
    model = MedicalResNet18(
        in_channels=config["dataset"]["in_channels"],
        num_classes=config["dataset"]["num_classes"],
        pretrained=False
    )

    # Train Baseline Model
    best_stats = train_centralized_model(model, config, device)

    # Evaluate Baseline Model on Test Set
    test_stats = run_baseline_evaluation(model, config, device)

    # Combine & Export Baseline Benchmarks for Month 4 Paper Comparison
    summary = {**best_stats, **test_stats}
    metrics_path = os.path.join(
        config["paths"]["checkpoint_dir"], config["paths"]["metrics_save_name"]
    )

    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\nSaved Centralized Baseline metrics summary to: {metrics_path}")


if __name__ == "__main__":
    main()