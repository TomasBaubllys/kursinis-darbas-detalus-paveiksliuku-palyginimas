# Author: Tomas Baublys
import torch
import torch.nn as nn
import torchvision.models as models

"""
class Siamese_Network(nn.Module):
    def __init__(self, num_classes=751):  # Market-1501 train set has 751 IDs
        super().__init__()
        self.backbone = models.resnet18(weights="DEFAULT")

        # Remove the original FC layer to access the 512-d features
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Embedding Head (for Triplet Loss)
        self.embedding_head = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),  # Features for Triplet Loss
        )

        # Classification Head (for Cross-Entropy Loss)
        # Note: Standard Re-ID practice uses the output of the first BatchNorm
        # or a separate linear layer for classification.
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.embedding_head(features)

        if self.training:
            cls_score = self.classifier(embeddings)
            return embeddings, cls_score

        return embeddings
"""


class Siamese_Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights="DEFAULT")

        # Freeze all backbone parameters
        # for param in self.backbone.parameters():
        #    param.requires_grad = False

        num_ftrs = self.backbone.fc.in_features

        # overwrite the top layer
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
        )

    def forward_once(self, x):
        return self.backbone(x)

    def forward(self, x):
        output = self.forward_once(x)
        return output
