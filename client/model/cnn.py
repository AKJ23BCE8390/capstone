import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class LocalClientCNN(nn.Module):
    """Lightweight 2-layer CNN for edge client verification (64x64 inputs)."""
    def __init__(self, in_channels: int = 1, num_classes: int = 2):
        super(LocalClientCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Spatial dimensions: 64x64 -> Pool1: 32x32 -> Pool2: 16x16
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MedicalResNet18(nn.Module):
    """ResNet-18 model adapted for medical image classification."""
    def __init__(self, in_channels: int = 1, num_classes: int = 2, pretrained: bool = False):
        super(MedicalResNet18, self).__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.model = models.resnet18(weights=weights)

        # Adjust initial Conv layer for grayscale images
        if in_channels != 3:
            self.model.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # Output FC layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# Legacy alias
BaselineResNet18 = MedicalResNet18