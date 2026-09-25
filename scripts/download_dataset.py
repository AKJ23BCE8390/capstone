import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from client.data.dataset import load_medmnist_splits


def main():
    print("Initiating PneumoniaMNIST dataset download...")
    train_ds, val_ds, test_ds = load_medmnist_splits()
    
    print("\n--- Dataset Loading Verification ---")
    print(f"Train Dataset Size: {len(train_ds)} samples")
    print(f"Validation Dataset Size: {len(val_ds)} samples")
    print(f"Test Dataset Size: {len(test_ds)} samples")

    # Save centralized .pt files required by run_centralized.py
    centralized_dir = "./data/centralized"
    os.makedirs(centralized_dir, exist_ok=True)

    torch.save({"images": train_ds.images, "labels": train_ds.labels}, os.path.join(centralized_dir, "train.pt"))
    torch.save({"images": val_ds.images, "labels": val_ds.labels}, os.path.join(centralized_dir, "val.pt"))
    torch.save({"images": test_ds.images, "labels": test_ds.labels}, os.path.join(centralized_dir, "test.pt"))

    print(f"\nSuccessfully saved serialized datasets to '{centralized_dir}/'")


if __name__ == "__main__":
    main()