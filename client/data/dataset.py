import os
import torch
from torch.utils.data import Dataset
from medmnist import PneumoniaMNIST
from client.data.preprocessing import get_data_transforms


class PneumoniaDataset(Dataset):
    def __init__(self, split="train", download=True, root="./data/raw"):
        train_tf, eval_tf = get_data_transforms(image_size=64)
        transform = train_tf if split == "train" else eval_tf

        self.medmnist_ds = PneumoniaMNIST(
            split=split, download=download, root=root, transform=transform
        )

        images = []
        labels = []
        for img, lbl in self.medmnist_ds:
            images.append(img)
            labels.append(torch.tensor(lbl, dtype=torch.long))

        self.images = torch.stack(images)
        self.labels = torch.stack(labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def load_medmnist_splits(data_dir="./data/raw"):
    os.makedirs(data_dir, exist_ok=True)
    train_ds = PneumoniaDataset(split="train", download=True, root=data_dir)
    val_ds = PneumoniaDataset(split="val", download=True, root=data_dir)
    test_ds = PneumoniaDataset(split="test", download=True, root=data_dir)
    return train_ds, val_ds, test_ds