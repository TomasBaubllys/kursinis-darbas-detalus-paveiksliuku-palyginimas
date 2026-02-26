# Author: Tomas Baublys
import torch
import torch.nn as nn
import torchvision.models as models


class Siamese_Network(nn.Module):
    def __init__(self, backbone="resnet18"):
        super().__init__()

        if backbone == "resnet18":
            self.backbone = models.resnet18(weights="DEFAULT")
        elif backbone == "resnet50":
            self.backbone = models.resnet50(weights="DEFAULT")

        # Freeze all backbone parameters
        # for param in self.backbone.parameters():
        #    param.requires_grad = False

        num_ftrs = self.backbone.fc.in_features

        # overwrite the top layer
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
        )

    def forward_once(self, x):
        self.backbone(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        return

    def forward(self, x):
        output = self.forward_once(x)
        return output
