import torch
import torch.nn as nn
from torchvision import models


class ResNet18_BoT(nn.Module):
    def __init__(self, num_classes, bot_level=0):
        super().__init__()
        resnet = models.resnet18(weights="DEFAULT")
        self.bot = bot_level
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.gap = nn.AdaptiveAvgPool2d(1)

        in_features = resnet.fc.in_features

        if bot_level == 0:
            self.bottleneck = nn.Identity()
            self.classifier = nn.Linear(in_features, num_classes)
            nn.init.normal_(self.classifier.weight, std=0.001)
            nn.init.constant_(self.classifier.bias, 0)
        if bot_level >= 1:
            self.bottleneck = nn.BatchNorm1d(in_features)
            self.bottleneck.bias.requires_grad_(False)  # BNNeck taisyklė
            self.classifier = nn.Linear(in_features, num_classes, bias=False)

        if bot_level >= 2:
            self.bottleneck.apply(self.weights_init_kaiming)
            self.classifier.apply(self.weights_init_classifier)

        if bot_level >= 3:
            resnet_layer4 = list(resnet.children())[7]
            resnet_layer4[0].conv1.stride = (1, 1)
            resnet_layer4[0].downsample[0].stride = (1, 1)
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])

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

        if self.bot == 0:
            if self.training:
                logits = self.classifier(global_feat)
                return global_feat, logits
            return global_feat
        bn_feat = self.bottleneck(global_feat)

        if self.training:
            cls_score = self.classifier(bn_feat)
            return global_feat, cls_score
        return bn_feat


class MobileNetV3_BoT(nn.Module):
    def __init__(self, num_classes, model_type="small", bot_level=0):
        super().__init__()

        if model_type == "large":
            mobilenet = models.mobilenet_v3_large(weights="DEFAULT")
        else:
            mobilenet = models.mobilenet_v3_small(weights="DEFAULT")

        self.backbone = mobilenet.features
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.bot = bot_level

        self.feat_dim = in_features = mobilenet.classifier[0].in_features

        if bot_level == 0:
            self.bottleneck = nn.Identity()
            self.classifier = mobilenet.classifier
            last_layer_idx = len(self.classifier) - 1
            self.classifier[last_layer_idx] = nn.Linear(
                self.classifier[last_layer_idx].in_features, num_classes
            )

        if bot_level >= 1:
            self.bottleneck = nn.BatchNorm1d(in_features)
            self.bottleneck.bias.requires_grad_(False)
            self.classifier = nn.Linear(in_features, num_classes, bias=False)

        if bot_level >= 2:
            self.bottleneck.apply(self.weights_init_kaiming)
            self.classifier.apply(self.weights_init_classifier)

        if bot_level >= 3:
            block_idx = 13 if model_type == "large" else 9
            for m in self.backbone[block_idx].modules():
                if isinstance(m, nn.Conv2d) and m.stride == (2, 2):
                    m.stride = (1, 1)

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
        if self.bot != 0:
            bn_feat = self.bottleneck(global_feat)
            if self.training:
                cls_score = self.classifier(bn_feat)
                return global_feat, cls_score
            return bn_feat

        if self.training:
            logits = self.classifier(global_feat)
            return global_feat, logits
        return global_feat
