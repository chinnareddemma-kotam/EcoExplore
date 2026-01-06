import torch.nn as nn
from torchvision import models


def build_model(model_name: str, num_classes: int, pretrained: bool = True):
    model_name = model_name.lower()

    if model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    else:
        raise ValueError("Supported models: resnet18, resnet50")

    return model


class SoilDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        return self.fc(x)


def build_soil_model():
    return SoilDetectionModel()
