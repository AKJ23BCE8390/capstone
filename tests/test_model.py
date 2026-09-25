import pytest
import torch
from client.model.cnn import LocalClientCNN, MedicalResNet18
from client.model.model_utils import verify_model_input_shape


def test_local_client_cnn_shape():
    device = torch.device("cpu")
    model = LocalClientCNN(in_channels=1, num_classes=2)
    # Batch size 4, 1 Channel, 64x64 Image
    valid = verify_model_input_shape(model, (4, 1, 64, 64), device)
    assert valid is True, "LocalClientCNN failed shape assertion for 64x64 inputs."


def test_medical_resnet18_shape():
    device = torch.device("cpu")
    model = MedicalResNet18(in_channels=1, num_classes=2)
    valid = verify_model_input_shape(model, (2, 1, 64, 64), device)
    assert valid is True, "MedicalResNet18 failed shape assertion for 64x64 inputs."