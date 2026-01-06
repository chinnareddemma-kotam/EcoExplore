import torch
import torch.nn as nn
from torchvision import models


class VegSegmentationModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )
        self.backbone.fc = nn.Identity()

        self.seg_head = nn.Conv2d(512, 1, kernel_size=1)

    def forward(self, x):
        features = self.backbone(x)       # (B, 512)
        features = features.view(-1, 512, 1, 1)
        return self.seg_head(features)


def build_veg_model(num_classes=2):
    import torch.nn as nn
    import torchvision.models as models

    model = models.segmentation.fcn_resnet50(pretrained=False)
    model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)  # make last layer match num_classes
    return model

