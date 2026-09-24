import numpy as np
import torch
from torch.utils.data import Subset

def partition_data_dirichlet(dataset, num_clients=4, alpha=0.5, seed=42):
    """
    Partitions a dataset across client nodes using a Dirichlet distribution Dir(alpha).
    Simulates non-IID label distribution skew across hospital clients.
    
    Args:
        dataset: PyTorch Dataset object
        num_clients: Total number of simulated hospital clients (default: 4)
        alpha: Concentration parameter for Dirichlet distribution (lower = more non-IID)
        seed: Random seed for reproducibility
        
    Returns:
        client_subsets: List of PyTorch Subset objects for each client
        client_indices: List of raw index lists per client
    """
    np.random.seed(seed)
    
    # Extract targets array regardless of dataset structure
    if hasattr(dataset, 'labels'):
        targets = np.array(dataset.labels)
    else:
        targets = np.array([y for _, y in dataset])
        
    if targets.ndim > 1:
        targets = targets.squeeze()

    num_classes = len(np.unique(targets))
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        idx_c = np.where(targets == c)[0]
        np.random.shuffle(idx_c)

        # Draw Dirichlet proportions for current class
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (proportions * len(idx_c)).astype(int)

        # Allocate residual samples to avoid rounding drops
        diff = len(idx_c) - proportions.sum()
        for i in range(diff):
            proportions[i % num_clients] += 1

        # Distribute indices to clients
        start = 0
        for client_id in range(num_clients):
            end = start + proportions[client_id]
            client_indices[client_id].extend(idx_c[start:end])
            start = end

    client_subsets = [Subset(dataset, indices) for indices in client_indices]
    return client_subsets, client_indices