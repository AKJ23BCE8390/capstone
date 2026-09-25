# tests/test_data_partition.py
import os
import unittest
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


class TestDataPartitioning(unittest.TestCase):
    
    def setUp(self):
        self.clients_dir = "./data/clients"
        self.num_clients = 4
        self.expected_hospitals = [f"hospital_{i+1}" for i in range(self.num_clients)]

    def test_hospital_directories_exist(self):
        """Verify that all 4 hospital client folders exist."""
        for hospital in self.expected_hospitals:
            hospital_path = os.path.join(self.clients_dir, hospital)
            self.assertTrue(os.path.exists(hospital_path), f"Missing folder: {hospital_path}")

    def test_partition_tensor_files_exist(self):
        """Verify that train.pt and val.pt exist for each hospital."""
        for hospital in self.expected_hospitals:
            train_pt = os.path.join(self.clients_dir, hospital, "train.pt")
            val_pt = os.path.join(self.clients_dir, hospital, "val.pt")
            self.assertTrue(os.path.exists(train_pt), f"Missing tensor: {train_pt}")
            self.assertTrue(os.path.exists(val_pt), f"Missing tensor: {val_pt}")

    def test_tensor_shapes_and_types(self):
        """Verify image tensor dimensions (N, 1, 64, 64) and label formats."""
        for hospital in self.expected_hospitals:
            train_data = torch.load(os.path.join(self.clients_dir, hospital, "train.pt"))
            images = train_data["images"]
            labels = train_data["labels"]

            self.assertEqual(images.ndim, 4, f"Images in {hospital} must be 4D tensors (N, C, H, W)")
            self.assertEqual(images.shape[1], 1, f"Image channel count in {hospital} must be 1 (grayscale)")
            self.assertEqual(images.shape[2], 64, f"Image height in {hospital} must be 64")
            self.assertEqual(images.shape[3], 64, f"Image width in {hospital} must be 64")
            self.assertGreater(len(labels), 0, f"Labels in {hospital} cannot be empty")

    def test_dataloader_batch_generation(self):
        """Verify DataLoader integration and batch creation for local client training."""
        for hospital in self.expected_hospitals:
            train_data = torch.load(os.path.join(self.clients_dir, hospital, "train.pt"))
            dataset = TensorDataset(train_data["images"], train_data["labels"])
            loader = DataLoader(dataset, batch_size=32, shuffle=True)

            batch_imgs, batch_lbls = next(iter(loader))
            self.assertEqual(batch_imgs.shape[0], min(32, len(dataset)))
            self.assertEqual(batch_imgs.shape[1:], (1, 64, 64))

    def test_non_iid_class_distribution(self):
        """Ensure Dirichlet partitioning created heterogeneous label distributions."""
        class_ratios = []
        for hospital in self.expected_hospitals:
            train_data = torch.load(os.path.join(self.clients_dir, hospital, "train.pt"))
            labels = train_data["labels"].numpy().squeeze()
            ratio_pos = np.mean(labels == 1)
            class_ratios.append(ratio_pos)

        # Variance across hospital class ratios should be non-zero for Non-IID Dirichlet splits
        std_dev = np.std(class_ratios)
        self.assertGreater(std_dev, 0.01, "Label distribution across clients is completely uniform (IID instead of Non-IID)")


if __name__ == "__main__":
    unittest.main()