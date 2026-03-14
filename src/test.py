import os
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupKFold

import download_data
from center_loss import Center_Loss
from cross_entropy_label_smooth import Cross_Entropy_Label_Smooth
from dataset import DEFAULT_DATA_PATH, Market_Train_Dataset
from models import MobileNetV3_BoT, ResNet18_BoT
from pksampler import PKSampler
from triplet_loss import Batch_Hard_Triplet_Loss

# Importing evaluation logic from your other file (assuming it's named evaluate_utils.py)
# from evaluate_utils import evaluate

RESNET18_NAME = "resnet18"
MOBILENETV3_NAME = "mobilenetv3"
CHECKPOINT_PATH = "../checkpoint"

def graph_loss(loss_hist, epochs, loss_names=["1", "2"], title=""):
    x = np.arange(epochs)
    plt.figure()
    for i, name in enumerate(loss_names):
        if i < len(loss_hist):
            data = [val.detach().cpu().item() if hasattr(val, 'cpu') else val for val in loss_hist[i]]
            plt.plot(x, data, label=name)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{title}")
    plt.legend()
    plt.savefig(f"{title}.jpg")
    plt.close()

def get_transforms(bot_level_train):
    if bot_level_train >= 1:
        return transforms.Compose([
            transforms.Resize((256, 128), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Pad(10),
            transforms.RandomCrop((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.4), ratio=(0.3, 3.33), value="random"),
        ])
    return transforms.Compose([
        transforms.Resize((256, 128), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Pad(10),
        transforms.RandomCrop((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def train_logic(dataloader, model_name, bot_level_model, bot_level_train, num_classes, save_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == MOBILENETV3_NAME:
        model = MobileNetV3_BoT(num_classes=num_classes, bot_level=bot_level_model).to(device)
    else:
        model = ResNet18_BoT(num_classes=num_classes).to(device)

    criterion_triplet = Batch_Hard_Triplet_Loss(margin=0.3).to(device)
    criterion_id = Cross_Entropy_Label_Smooth(num_classes=num_classes).to(device)

    feat_dim = model.feat_dim if hasattr(model, 'feat_dim') else 512
    criterion_center = Center_Loss(num_classes=num_classes, feat_dim=feat_dim).to(device)

    center_loss_weight = 0.0005
    optimizer = optim.Adam(model.parameters(), lr=3.5e-4, weight_decay=5e-4)
    center_optimizer = optim.SGD(criterion_center.parameters(), lr=0.5)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda e: 1.0 if e < 40 else (0.1 if e < 70 else 0.01))

    num_epochs = 120
    loss_hist = [[], [], [], []]

    model.train()
    for epoch in range(num_epochs):
        running_loss, running_id, running_tri, running_center = 0.0, 0.0, 0.0, 0.0
        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            embeddings, logits = model(images)

            loss_triplet = criterion_triplet(embeddings, labels)
            loss_id = criterion_id(logits, labels)
            loss_center = criterion_center(embeddings, labels)

            total_loss = loss_id + loss_triplet
            if bot_level_train >= 2:
                total_loss += center_loss_weight * loss_center

            optimizer.zero_grad()
            center_optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if bot_level_train >= 2:
                for param in criterion_center.parameters():
                    param.grad.data *= (1.0 / center_loss_weight)
                center_optimizer.step()

            running_loss += total_loss.item()
            running_id += loss_id.item()
            running_tri += loss_triplet.item()
            running_center += loss_center.item()

        scheduler.step()
        loss_hist[0].append(running_loss / len(dataloader))
        loss_hist[1].append(running_id / len(dataloader))
        loss_hist[2].append(running_tri / len(dataloader))
        loss_hist[3].append(running_center / len(dataloader))

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss_hist[0][-1]:.4f}")

    torch.save(model.state_dict(), save_name)
    return loss_hist

def run_kfold_training(args):
    # 1. Load data and setup GroupKFold
    # We load without transforms first to get the structure
    full_dataset = Market_Train_Dataset(DEFAULT_DATA_PATH)

    # Person IDs are our "Groups" to prevent data leakage
    pids = np.array(full_dataset.ids)
    indices = np.arange(len(full_dataset))

    gkf = GroupKFold(n_splits=5)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(indices, groups=pids)):
        print(f"\n{'='*30}\nWORKING ON FOLD {fold+1}\n{'='*30}")

        # Create subsets
        train_subset = Subset(full_dataset, train_idx)

        # Apply transforms to the underlying dataset (be careful if sharing dataset object)
        full_dataset.transform = get_transforms(args.bot_train)

        # Re-init sampler for the specific IDs in this fold
        p_val, k_val = 16, 4
        sampler = PKSampler(train_subset, p=p_val, k=k_val)
        dataloader = DataLoader(train_subset, sampler=sampler, num_workers=4, batch_size=p_val*k_val)

        save_path = os.path.join(CHECKPOINT_PATH, f"fold_{fold}_{args.model}_weights.pth")

        loss = train_logic(
            dataloader=dataloader,
            model_name=args.model,
            bot_level_model=args.bot_model,
            bot_level_train=args.bot_train,
            num_classes=full_dataset.num_ids,
            save_name=save_path
        )

        graph_loss(loss, len(loss[0]), ["Total", "ID", "Triplet", "Center"], f"Fold_{fold}_Loss")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process model settings.")
    parser.add_argument("-red", "--refresh_data", action="store_true")
    parser.add_argument("-pltls", "--plot_loss", action="store_true")
    parser.add_argument("-tgs", "--train_grid_search", action="store_true")
    parser.add_argument("-kf", "--kfold", action="store_true", help="Run 5-Fold Cross Validation")
    parser.add_argument("-bm", "--bot_model", type=int, default=1)
    parser.add_argument("-bt", "--bot_train", type=int, default=1)
    parser.add_argument("-mbnet", "--mobilenet", action="store_const", const="mobilenetv3", dest="model", default="resnet18")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    if args.refresh_data:
        download_data.setup_market1501()

    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)

    if args.kfold:
        run_kfold_training(args)
    else:
        # Standard training logic
        ds = Market_Train_Dataset(DEFAULT_DATA_PATH, transform=get_transforms(args.bot_train))
        sampler = PKSampler(ds, p=16, k=4)
        loader = DataLoader(ds, sampler=sampler, num_workers=4, batch_size=64)

        loss = train_logic(
            dataloader=loader,
            model_name=args.model,
            bot_level_model=args.bot_model,
            bot_level_train=args.bot_train,
            num_classes=ds.num_ids,
            save_name=f"{args.model}_final.pth"
        )
        graph_loss(loss, len(loss[0]), ["Total", "ID", "Triplet", "Center"], "Final_Training_Loss")
