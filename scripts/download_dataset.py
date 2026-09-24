import os
import sys

# Ensure repository root is in Python execution path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from client.data.dataset import load_medmnist_splits

def main():
    print("Initiating PneumoniaMNIST dataset download...")
    train_ds, val_ds, test_ds = load_medmnist_splits()
    print("\n--- Dataset Loading Verification ---")
    print(f"Train Dataset Size: {len(train_ds)} samples")
    print(f"Validation Dataset Size: {len(val_ds)} samples")
    print(f"Test Dataset Size: {len(test_ds)} samples")

if __name__ == "__main__":
    main()