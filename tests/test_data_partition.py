import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from client.data.dataset import load_medmnist_splits
from client.data.partition import partition_data_dirichlet

def test_partition_integrity():
    train_dataset, _, _ = load_medmnist_splits()
    total_samples = len(train_dataset)
    num_clients = 4
    
    client_subsets, client_indices = partition_data_dirichlet(
        dataset=train_dataset,
        num_clients=num_clients,
        alpha=0.5,
        seed=42
    )

    # Assert correct client count
    assert len(client_subsets) == num_clients, f"Expected {num_clients} client subsets."

    # Assert no missing or duplicated indices
    all_indices = []
    for indices in client_indices:
        all_indices.extend(indices)
        
    assert len(all_indices) == total_samples, "Total partitioned samples do not match raw dataset size."
    assert len(set(all_indices)) == total_samples, "Duplicate sample indices detected across client partitions."

    print("✅ Non-IID Dirichlet Partition Unit Tests Passed!")

if __name__ == "__main__":
    test_partition_integrity()