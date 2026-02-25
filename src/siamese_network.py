# Author: Tomas Baublys

import torch
import torch.nn as nn
import torchvision.models as models


class Siamese_Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights="DEFAULT")

        # Freeze all backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        num_ftrs = self.backbone.fc.in_features

        # overwrite the top layer
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256), nn.ReLU(), nn.Linear(256, 128)
        )

    def forward_once(self, x):
        return self.backbone(x)

    def forward(self, x):
        output = self.forward_once(x)
        return output
