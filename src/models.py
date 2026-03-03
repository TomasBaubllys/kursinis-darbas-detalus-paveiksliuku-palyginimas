import torch
import torch.nn as nn
from torchvision import models


class ResNet18_BoT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Load pre-trained ResNet18
        resnet = models.resnet18(weights="DEFAULT")

        # Increasing the spatial resolution from 8x4 to 16x8 for richer features
        resnet.layer4[0].conv1.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)

        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.gap = nn.AdaptiveAvgPool2d(1)

        in_features = resnet.fc.in_features

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

        feat_for_triplet = global_feat

        bn_feat = self.bottleneck(global_feat)

        if self.training:
            cls_score = self.classifier(bn_feat)
            return feat_for_triplet, cls_score
        else:
            return bn_feat


class MobileNetV3_BoT(nn.Module):
    def __init__(self, num_classes, model_type="small", bot=False):
        super().__init__()

        if model_type == "large":
            mobilenet = models.mobilenet_v3_large(weights="DEFAULT")
        else:
            mobilenet = models.mobilenet_v3_small(weights="DEFAULT")

        self.backbone = mobilenet.features
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.bot = bot

        self.feat_dim = in_features = mobilenet.classifier[0].in_features

        if bot:
            self.bottleneck = nn.BatchNorm1d(in_features)
            self.bottleneck.bias.requires_grad_(False)
            self.bottleneck.apply(self.weights_init_kaiming)

            self.classifier = nn.Linear(in_features, num_classes, bias=False)
            self.classifier.apply(self.weights_init_classifier)
        else:
            self.bottleneck = nn.Identity()
            self.classifier = mobilenet.classifier
            last_layer_idx = len(self.classifier) - 1
            self.classifier[last_layer_idx] = nn.Linear(
                self.classifier[last_layer_idx].in_features, num_classes
            )

    def weights_init_kaiming(self, m):
        if isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def weights_init_classifier(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.001)

    def forward(self, x):
        x = self.backbone(x)
        x = self.gap(x)
        global_feat = x.view(x.shape[0], -1)

        if self.bot:
            bn_feat = self.bottleneck(global_feat)
            if self.training:
                cls_score = self.classifier(bn_feat)
                return global_feat, cls_score
            else:
                return bn_feat
        else:
            logits = self.classifier(global_feat)
            if self.training:
                return global_feat, logits
            return logits
