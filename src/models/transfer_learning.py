import torch
import torch.nn as nn
from torchvision import models


def create_transfer_learning_model(num_classes=1, freeze_layers=True, dropout_rate=0.5):
    """
    Create a transfer learning model based on ResNet18

    Args:
        num_classes: Number of output classes (1 for binary)
        freeze_layers: Whether to freeze early layers
        dropout_rate: Dropout rate for the final layers

    Returns:
        PyTorch model with transfer learning
    """
    # Load pre-trained ResNet18
    model = models.resnet18(pretrained=True)

    # Freeze early layers if specified
    if freeze_layers:
        # Freeze all but the last few layers
        for param in list(model.parameters())[:-8]:
            param.requires_grad = False

    # Get number of features in the final layer
    num_features = model.fc.in_features

    # Replace final layer for classification
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(512, num_classes),
        nn.Sigmoid() if num_classes == 1 else nn.LogSoftmax(dim=1)
    )

    return model