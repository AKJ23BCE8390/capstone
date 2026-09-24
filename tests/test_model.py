import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from client.model.cnn import LocalClientCNN, BaselineResNet18
from client.model.model_utils import count_parameters

def test_architectures():
    dummy_input = torch.randn(4, 1, 64, 64) # Batch size: 4, Channels: 1, Height: 64, Width: 64

    # Test Lightweight Local CNN
    local_cnn = LocalClientCNN(in_channels=1, num_classes=2)
    out_local = local_cnn(dummy_input)
    assert out_local.shape == (4, 2), f"Expected shape (4, 2), received {out_local.shape}"
    print(f"LocalClientCNN Output Shape: {out_local.shape} | Parameters: {count_parameters(local_cnn):,}")

    # Test Baseline ResNet-18
    resnet = BaselineResNet18(num_classes=2, pretrained=False)
    out_resnet = resnet(dummy_input)
    assert out_resnet.shape == (4, 2), f"Expected shape (4, 2), received {out_resnet.shape}"
    print(f"BaselineResNet18 Output Shape: {out_resnet.shape} | Parameters: {count_parameters(resnet):,}")

    print("\n✅ All Model Architecture Unit Tests Passed!")

if __name__ == "__main__":
    test_architectures()