# scripts/create_non_iid.py
import os
import sys
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from client.data.dataset import load_medmnist_splits
from client.data.partition import partition_data_dirichlet


def main():
    print("Generating Non-IID Dirichlet Data Splits (alpha=0.5, 4 hospital clients)...")
    train_dataset, val_dataset, _ = load_medmnist_splits()
    
    # 1. Partition Train Dataset
    _, train_indices = partition_data_dirichlet(
        dataset=train_dataset,
        num_clients=4,
        alpha=0.5,
        seed=42
    )

    # 2. Partition Validation Dataset
    _, val_indices = partition_data_dirichlet(
        dataset=val_dataset,
        num_clients=4,
        alpha=0.5,
        seed=42
    )

    print("\n--- Non-IID Class Distribution per Hospital Client ---")
    clients_base_dir = "./data/clients"

    for client_id in range(4):
        t_indices = train_indices[client_id]
        v_indices = val_indices[client_id]

        targets = np.array([train_dataset[i][1].item() for i in t_indices])
        class_0_count = np.sum(targets == 0)  # Normal
        class_1_count = np.sum(targets == 1)  # Pneumonia
        total = len(t_indices)
        
        ratio_0 = (class_0_count / total) * 100
        ratio_1 = (class_1_count / total) * 100
        
        print(f"Hospital {client_id + 1}: Total={total:4d} | "
              f"Normal (0)={class_0_count:4d} ({ratio_0:5.1f}%) | "
              f"Pneumonia (1)={class_1_count:4d} ({ratio_1:5.1f}%)")

        # Save hospital partition tensors
        hospital_dir = os.path.join(clients_base_dir, f"hospital_{client_id + 1}")
        os.makedirs(hospital_dir, exist_ok=True)
        
        # Save train.pt
        client_train_imgs = train_dataset.images[t_indices]
        client_train_lbls = train_dataset.labels[t_indices]
        torch.save({"images": client_train_imgs, "labels": client_train_lbls}, os.path.join(hospital_dir, "train.pt"))

        # Save val.pt
        client_val_imgs = val_dataset.images[v_indices]
        client_val_lbls = val_dataset.labels[v_indices]
        torch.save({"images": client_val_imgs, "labels": client_val_lbls}, os.path.join(hospital_dir, "val.pt"))

    print(f"\nSuccessfully generated and saved Non-IID train.pt and val.pt to '{clients_base_dir}/'")


if __name__ == "__main__":
    main()