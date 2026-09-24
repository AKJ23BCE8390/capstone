import os
import medmnist
from medmnist import INFO
from client.data.preprocessing import get_data_transforms

def load_medmnist_splits(dataset_name="pneumoniamnist", image_size=64, root_dir="./data/raw"):
    """
    Downloads and loads the PneumoniaMNIST train, validation, and test dataset splits.
    """
    os.makedirs(root_dir, exist_ok=True)
    info = INFO[dataset_name]
    DataClass = getattr(medmnist, info['python_class'])

    train_tf, eval_tf = get_data_transforms(image_size=image_size)

    train_dataset = DataClass(split='train', transform=train_tf, download=True, root=root_dir)
    val_dataset = DataClass(split='val', transform=eval_tf, download=True, root=root_dir)
    test_dataset = DataClass(split='test', transform=eval_tf, download=True, root=root_dir)

    return train_dataset, val_dataset, test_dataset