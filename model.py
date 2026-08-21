"""
Shared CNN architecture for both ASL and ISL sign language models.
Uses MobileNetV2 pretrained on ImageNet with a custom classifier head.
"""
import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Build a MobileNetV2-based classifier for sign language recognition.

    Args:
        num_classes: Number of output classes (29 for ASL, 35 for ISL).
        pretrained: Whether to load ImageNet pretrained weights.

    Returns:
        A PyTorch model ready for fine-tuning or inference.
    """
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)

    # Freeze all backbone layers initially
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last 3 InvertedResidual blocks + classifier for fine-tuning
    trainable_layers = list(model.features.children())[-4:]
    for layer in trainable_layers:
        for param in layer.parameters():
            param.requires_grad = True

    # Replace the classifier head
    in_features = model.classifier[1].in_features  # 1280 for MobileNetV2
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes),
    )

    return model


def load_model(checkpoint_path: str, num_classes: int, device: torch.device) -> nn.Module:
    """
    Load a trained model from a checkpoint.

    Args:
        checkpoint_path: Path to the .pth file.
        num_classes: Number of output classes.
        device: torch device to load onto.

    Returns:
        Model in eval mode on the specified device.
    """
    model = build_model(num_classes=num_classes, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"[✓] Loaded model from {checkpoint_path} | Classes: {num_classes} | Device: {device}")
    return model


# ImageNet normalization constants (used for both train & inference)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD  = [0.229, 0.224, 0.225]
INPUT_SIZE     = 224
