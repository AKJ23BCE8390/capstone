import os
import sys
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from client.data.dataset import load_medmnist_splits
from client.data.partition import partition_data_dirichlet


def main():
    print("Generating Non-IID Dirichlet Data Splits (alpha=0.5, 4 clients)...")
    train_dataset, _, _ = load_medmnist_splits()
    
    client_subsets, client_indices = partition_data_dirichlet(
        dataset=train_dataset,
        num_clients=4,
        alpha=0.5,
        seed=42
    )

    print("\n--- Non-IID Class Distribution per Hospital Client ---")
    clients_base_dir = "./data/clients"

    for client_id, indices in enumerate(client_indices):
        targets = np.array([train_dataset[i][1].item() for i in indices])
        class_0_count = np.sum(targets == 0) # Normal
        class_1_count = np.sum(targets == 1) # Pneumonia
        total = len(indices)
        
        ratio_0 = (class_0_count / total) * 100
        ratio_1 = (class_1_count / total) * 100
        
        print(f"Hospital {client_id + 1}: Total={total:4d} | "
              f"Normal (0)={class_0_count:4d} ({ratio_0:5.1f}%) | "
              f"Pneumonia (1)={class_1_count:4d} ({ratio_1:5.1f}%)")

        # Save hospital partition tensors
        hospital_dir = os.path.join(clients_base_dir, f"hospital_{client_id + 1}")
        os.makedirs(hospital_dir, exist_ok=True)
        
        client_images = train_dataset.images[indices]
        client_labels = train_dataset.labels[indices]
        torch.save({"images": client_images, "labels": client_labels}, os.path.join(hospital_dir, "train.pt"))

    print(f"\nSuccessfully generated and saved Non-IID hospital datasets to '{clients_base_dir}/'")


if __name__ == "__main__":
    main()