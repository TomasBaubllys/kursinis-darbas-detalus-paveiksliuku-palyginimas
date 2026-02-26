import torch
import torch.nn as nn
from torchvision import models


class Siamese_Network(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Load pre-trained ResNet18
        resnet = models.resnet18(weights="DEFAULT")

        # TRICK 1: Last Stride Trick
        # Increasing the spatial resolution from 8x4 to 16x8 for richer features
        resnet.layer4[0].conv1.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)

        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.gap = nn.AdaptiveAvgPool2d(1)

        in_features = resnet.fc.in_features

        # TRICK 2: BNNeck (Batch Normalization Neck)
        # Separates metric space and classification space
        self.bottleneck = nn.BatchNorm1d(in_features)
        self.bottleneck.bias.requires_grad_(False)  # No bias for BNNeck
        self.bottleneck.apply(self.weights_init_kaiming)

        # Classifier (No bias as per the paper)
        self.classifier = nn.Linear(in_features, num_classes, bias=False)
        self.classifier.apply(self.weights_init_classifier)

    def weights_init_kaiming(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def weights_init_classifier(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.001)

    def forward(self, x):
        global_feat = self.gap(self.backbone(x))
        global_feat = global_feat.view(global_feat.shape[0], -1)

        # Metric Loss (Triplet) uses features BEFORE BNNeck
        feat_for_triplet = global_feat

        # Classification Loss (ID) uses features AFTER BNNeck
        bn_feat = self.bottleneck(global_feat)

        if self.training:
            cls_score = self.classifier(bn_feat)
            return feat_for_triplet, cls_score
        else:
            # Paper suggests using bn_feat for evaluation
            return bn_feat
