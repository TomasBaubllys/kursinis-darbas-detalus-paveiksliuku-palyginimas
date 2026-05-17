import argparse
import os
import sys
from math import dist

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from scipy.spatial.distance import cdist
from sklearn.model_selection import KFold
from torch.types import Device
from torch.utils.data import DataLoader

import download_data
from center_loss import Center_Loss
from cross_entropy_label_smooth import Cross_Entropy_Label_Smooth
from dataset import DEFAULT_DATA_PATH, Market_Train_Dataset
from models import MobileNetV3_BoT, ResNet18_BoT
from pksampler import PKSampler
from test import compute_metrics, extract_features
from triplet_loss import Batch_Hard_Triplet_Loss

RESNET18_NAME = "resnet18"
MOBILENETV3_NAME = "mobilenetv3"
CHECKPOINT_PATH = "../checkpoint"
WEIGHTS_PATH = "./weights/"


def graph_loss(loss_hist, epochs, loss_names=["1", "2"], title=""):
    """Plot multiple loss curves and save to disk."""
    x = np.arange(1, epochs + 1)
    plt.figure(figsize=(10, 5))
    for i, name in enumerate(loss_names):
        data = [
            val.detach().cpu().item() if hasattr(val, "cpu") else val
            for val in loss_hist[i]
        ]
        plt.plot(x, data, label=name)

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{title}.jpg")
    plt.close()
    print(f"  [Plot saved] {title}.jpg")


def plot_fold_summary(
    all_fold_train_losses, all_fold_val_losses, all_fold_ranks, num_epochs, model_name
):
    x = np.arange(1, num_epochs + 1)
    n_folds = len(all_fold_train_losses)
    loss_labels = ["Avg Loss", "ID Loss", "Triplet Loss", "Center Loss"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{model_name} – Training Loss per Fold", fontsize=14)
    for li, (ax, label) in enumerate(zip(axes.flat, loss_labels)):
        # Calculate matrix to get mean
        train_matrix = np.array(
            [
                [
                    v.detach().cpu().item() if hasattr(v, "cpu") else v
                    for v in fold_hist[li]
                ]
                for fold_hist in all_fold_train_losses
            ]
        )

        # Plot individual folds (lighter)
        for fold_idx in range(n_folds):
            ax.plot(
                x,
                train_matrix[fold_idx],
                label=f"Fold {fold_idx + 1}",
                alpha=0.5,
                linewidth=1,
            )

        # Plot Average (thicker)
        ax.plot(
            x,
            train_matrix.mean(axis=0),
            label="TRAIN AVG",
            color="black",
            linewidth=2,
            linestyle="--",
        )

        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.savefig(f"{model_name}_train_per_fold.jpg")
    plt.close()

    # ---- 2. Per-fold validation curves with Average ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{model_name} – Validation Loss per Fold", fontsize=14)
    for li, (ax, label) in enumerate(zip(axes.flat, loss_labels)):
        val_matrix = np.array(
            [
                [
                    v.detach().cpu().item() if hasattr(v, "cpu") else v
                    for v in fold_hist[li]
                ]
                for fold_hist in all_fold_val_losses
            ]
        )

        for fold_idx in range(n_folds):
            ax.plot(
                x,
                val_matrix[fold_idx],
                label=f"Fold {fold_idx + 1}",
                alpha=0.5,
                linewidth=1,
            )

        # Plot Average (thicker)
        ax.plot(
            x,
            val_matrix.mean(axis=0),
            label="Average",
            color="black",
            linewidth=2,
            linestyle="--",
        )

        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.savefig(f"{model_name}_val_per_fold.jpg")
    plt.close()

    # mean and std
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{model_name} – Mean ± Std across {n_folds} Folds", fontsize=14)
    for li, (ax, label) in enumerate(zip(axes.flat, loss_labels)):
        # train
        train_matrix = np.array(
            [
                [
                    v.detach().cpu().item() if hasattr(v, "cpu") else v
                    for v in fold_hist[li]
                ]
                for fold_hist in all_fold_train_losses
            ]
        )  # shape: (n_folds, num_epochs)WEIGHTS_PATH
        t_mean = train_matrix.mean(axis=0)
        t_std = train_matrix.std(axis=0)
        ax.plot(x, t_mean, label="Train mean", color="tab:blue")
        ax.fill_between(x, t_mean - t_std, t_mean + t_std, alpha=0.2, color="tab:blue")

        # val
        val_matrix = np.array(
            [
                [
                    v.detach().cpu().item() if hasattr(v, "cpu") else v
                    for v in fold_hist[li]
                ]
                for fold_hist in all_fold_val_losses
            ]
        )
        v_mean = val_matrix.mean(axis=0)
        v_std = val_matrix.std(axis=0)
        ax.plot(x, v_mean, label="Val mean", color="tab:orange")
        ax.fill_between(
            x, v_mean - v_std, v_mean + v_std, alpha=0.2, color="tab:orange"
        )

        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8)
    plt.tight_layout()
    out = f"{model_name}_mean_std_summary.jpg"
    plt.savefig(out)
    plt.close()
    print(f"  [Plot saved] {out}")

    rank_labels = ["Rank-R1", "Rank-R5", "mAP"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"{model_name} – Validation Rank per Fold", fontsize=14)
    for li, (ax, label) in enumerate(zip(axes.flat, rank_labels)):
        val_matrix = np.array(
            [
                [
                    v.detach().cpu().item() if hasattr(v, "cpu") else v
                    for v in fold_hist[li]
                ]
                for fold_hist in all_fold_ranks
            ]
        )

        for fold_idx in range(n_folds):
            ax.plot(
                x,
                val_matrix[fold_idx],
                label=f"Fold {fold_idx + 1}",
                alpha=0.5,
                linewidth=1,
            )

        # Plot Average (thicker)
        ax.plot(
            x,
            val_matrix.mean(axis=0),
            label="Average",
            color="black",
            linewidth=2,
            linestyle="--",
        )

        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Rank, %")
        ax.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    plt.savefig(f"{model_name}_val_rank_per_fold.jpg")
    plt.close()

    print(f"  [Plot saved] {out}")


def plot_single_fold(fold_train_hist, fold_val_hist, num_epochs, title):
    """Train vs Val for a single fold – four subplots (one per loss type)."""
    x = np.arange(1, num_epochs + 1)
    loss_labels = ["Avg Loss", "ID Loss", "Triplet Loss", "Center Loss"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14)
    for li, (ax, label) in enumerate(zip(axes.flat, loss_labels)):
        train_data = [
            v.detach().cpu().item() if hasattr(v, "cpu") else v
            for v in fold_train_hist[li]
        ]
        val_data = [
            v.detach().cpu().item() if hasattr(v, "cpu") else v
            for v in fold_val_hist[li]
        ]
        ax.plot(x, train_data, label="Train", color="tab:blue")
        ax.plot(x, val_data, label="Val", color="tab:orange")
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
    plt.tight_layout()
    out = f"{title}.jpg"
    plt.savefig(out)
    plt.close()
    print(f"  [Plot saved] {out}")


def calc_one_epoch_r1_map(model: nn.Module, val_loader: DataLoader, device: Device):
    model.eval()
    all_feats, all_pids, all_camids = extract_features(model, val_loader, device)
    dist_mat = cdist(all_feats.numpy(), all_feats.numpy(), metric="cosine")
    return compute_metrics(
        dist_mat, all_pids, all_pids, all_camids, all_camids, include_same=False
    )


def validate_one_epoch(
    model,
    val_loader,
    criterion_triplet,
    criterion_id,
    criterion_center,
    center_loss_weight,
    bot_level_train,
    device,
):
    # Keep train() mode so output signature stays (embeddings, logits)
    model.train()
    running_loss = running_id = running_triplet = running_center = 0.0

    with torch.no_grad():
        for images, labels, _ in val_loader:
            images, labels = images.to(device), labels.to(device)
            out = model(images)
            # Defensively unpack: take only first two elements in case the
            # model returns extra tensors in certain configurations.
            embeddings, logits = out[0], out[1]

            loss_triplet = criterion_triplet(embeddings, labels)
            loss_id = criterion_id(logits, labels)
            loss_center = criterion_center(embeddings, labels)

            total = loss_id + loss_triplet
            if bot_level_train >= 2:
                total += center_loss_weight * loss_center
                running_center += (center_loss_weight * loss_center).item()

            running_loss += total.item()
            running_id += loss_id.item()
            running_triplet += loss_triplet.item()

    n = len(val_loader)
    return (
        running_loss / n,
        running_id / n,
        running_triplet / n,
        running_center / n,
    )


# transformations
def get_transformations(bot_level_train):
    if bot_level_train >= 1:
        return transforms.Compose(
            [
                transforms.Resize(
                    (224, 224), interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Pad(10),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                transforms.RandomErasing(
                    p=0.5, scale=(0.02, 0.4), ratio=(0.3, 3.33), value="random"
                ),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize(
                    (256, 128), interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.Pad(10),
                transforms.RandomCrop((256, 128)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )


def get_val_transformations():
    return transforms.Compose(
        [
            transforms.Resize(
                (256, 128), interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def train(
    refresh_data=False,
    plot_loss=False,
    save_name="checkpoint.pth",
    model_name=RESNET18_NAME,
    bot_level_model=0,
    bot_level_train=0,
    lr_function=1,
    num_epochs=120,
):
    if refresh_data:
        download_data.setup_market1501()

    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transformations = get_transformations(bot_level_train)

    market_dataset = Market_Train_Dataset(DEFAULT_DATA_PATH, transform=transformations)
    num_classes = market_dataset.num_ids

    p_val, k_val = 16, 4
    market_sampler = PKSampler(market_dataset, p=p_val, k=k_val)
    dataloader = DataLoader(
        market_dataset, sampler=market_sampler, num_workers=4, batch_size=p_val * k_val
    )

    if model_name == MOBILENETV3_NAME:
        model = MobileNetV3_BoT(num_classes=num_classes, bot_level=bot_level_model).to(
            device
        )
    elif model_name == RESNET18_NAME:
        model = ResNet18_BoT(num_classes=num_classes, bot_level=bot_level_model).to(
            device
        )

    print(model)

    criterion_triplet = Batch_Hard_Triplet_Loss(margin=0.3).to(device)

    if bot_level_train >= 2:
        criterion_id = Cross_Entropy_Label_Smooth(num_classes=num_classes).to(device)
    else:
        criterion_id = nn.CrossEntropyLoss().to(device)

    if model_name == MOBILENETV3_NAME:
        criterion_center = Center_Loss(
            num_classes=num_classes, feat_dim=model.feat_dim
        ).to(device)
    elif model_name == RESNET18_NAME:
        criterion_center = Center_Loss(num_classes=num_classes).to(device)

    center_loss_weight = 0.0005

    optimizer = optim.Adam(model.parameters(), lr=3.5e-4, weight_decay=5e-4)
    center_optimizer = optim.SGD(criterion_center.parameters(), lr=0.5)

    # no warmup
    def lr_lambda0(epoch):
        if epoch < 40:
            return 1
        elif epoch < 70:
            return 0.1
        else:
            return 0.01

    # original from the paper
    def lr_lambda1(epoch):
        if epoch < 10:
            return (epoch + 1) / 10
        if epoch < 40:
            return 1
        elif epoch < 70:
            return 0.1
        else:
            return 0.01

    def lr_lambda2(epoch):
        if epoch < 150:
            return 1.0
        elif epoch < 180:
            return 0.1
        else:
            return 0.01

    if lr_function == 0:
        lr_lambda = lr_lambda0
    elif lr_function == 1:
        lr_lambda = lr_lambda1
    else:
        lr_lambda = lr_lambda2

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    loss_hist = [[], [], [], []]

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_id_loss = 0.0
        running_triplet_loss = 0.0
        running_center_loss = 0.0
        for i, (images, labels, _) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)

            embeddings, logits = model(images)

            loss_triplet = criterion_triplet(embeddings, labels)
            loss_id = criterion_id(logits, labels)
            loss_center = criterion_center(embeddings, labels)

            running_id_loss += loss_id
            running_triplet_loss += loss_triplet
            total_loss = loss_id + loss_triplet
            if bot_level_train >= 2:
                total_loss += center_loss_weight * loss_center
                running_center_loss += center_loss_weight * loss_center

            optimizer.zero_grad()
            center_optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            if (i + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {total_loss.item():.4f} (ID: {loss_id.item():.2f}, Trp: {loss_triplet.item():.2f})"
                )

            if bot_level_train >= 3:
                for param in criterion_center.parameters():
                    param.grad.data *= 1.0 / center_loss_weight
                center_optimizer.step()

        scheduler.step()
        avg_loss = running_loss / len(dataloader)
        loss_hist[0].append(avg_loss)
        loss_hist[1].append(running_id_loss / len(dataloader))
        loss_hist[2].append(running_triplet_loss / len(dataloader))
        loss_hist[3].append(running_center_loss / len(dataloader))

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1} complete. LR: {current_lr:.6f}, Avg Loss: {avg_loss:.4f}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join(CHECKPOINT_PATH, f"checkpoint_{epoch}.pth"))

    torch.save(model.state_dict(), save_name)
    if plot_loss:
        graph_loss(
            loss_hist,
            num_epochs,
            loss_names=["Avg Loss", "ID Loss", "Triplet Loss", "Center Loss"],
            title=f"{model_name}_loss",
        )
    print("Training Finished")
    return loss_hist


def train_kfold(
    refresh_data=False,
    plot_loss=False,
    save_name="checkpoint.pth",
    model_name=RESNET18_NAME,
    bot_level_model=0,
    bot_level_train=0,
    lr_function=1,
    num_epochs=120,
):
    if refresh_data:
        download_data.setup_market1501()

    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    full_dataset = Market_Train_Dataset(DEFAULT_DATA_PATH)
    all_pids = np.array(full_dataset.person_ids)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    all_fold_train_losses = []
    all_fold_val_losses = []
    all_fold_ranks = []

    train_transforms = get_transformations(bot_level_train)
    val_transforms = get_val_transformations()

    for fold, (train_pid_idx, val_pid_idx) in enumerate(kf.split(all_pids)):
        print(f"\n{'=' * 60}")
        print(f"  FOLD {fold + 1} / 5")
        print(f"{'=' * 60}")

        train_pids = list(all_pids[train_pid_idx])
        val_pids = list(all_pids[val_pid_idx])

        train_dataset = Market_Train_Dataset(
            DEFAULT_DATA_PATH, transform=train_transforms, id_list=train_pids
        )
        val_dataset = Market_Train_Dataset(
            DEFAULT_DATA_PATH, transform=val_transforms, id_list=val_pids
        )

        num_classes = train_dataset.num_ids
        print(
            f"  Train PIDs: {len(train_pids)} | Val PIDs: {len(val_pids)} | Classes: {num_classes}"
        )

        p_val, k_val = 16, 4
        train_sampler = PKSampler(train_dataset, p=p_val, k=k_val)
        train_loader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            num_workers=4,
            batch_size=p_val * k_val,
        )

        val_p = min(p_val, len(val_pids))
        val_sampler = PKSampler(val_dataset, p=val_p, k=k_val)
        val_loader = DataLoader(
            val_dataset, sampler=val_sampler, num_workers=4, batch_size=val_p * k_val
        )

        if model_name == MOBILENETV3_NAME:
            model = MobileNetV3_BoT(
                num_classes=num_classes, bot_level=bot_level_model
            ).to(device)
        elif model_name == RESNET18_NAME:
            model = ResNet18_BoT(num_classes=num_classes, bot_level=bot_level_model).to(
                device
            )
        else:
            raise ValueError("Unrecognized model!")

        criterion_triplet = Batch_Hard_Triplet_Loss(margin=0.3).to(device)
        if bot_level_train >= 2:
            criterion_id = Cross_Entropy_Label_Smooth(num_classes=num_classes).to(
                device
            )
        else:
            criterion_id = nn.CrossEntropyLoss().to(device)

        if model_name == MOBILENETV3_NAME:
            criterion_center = Center_Loss(
                num_classes=num_classes, feat_dim=model.feat_dim
            ).to(device)
        elif model_name == RESNET18_NAME:
            criterion_center = Center_Loss(num_classes=num_classes).to(device)

        center_loss_weight = 0.0005

        # optimizer = optim.Adam(model.parameters(), lr=3.5e-4, weight_decay=5e-4)
        optimizer = optim.Adam(model.parameters(), lr=3.5e-4, weight_decay=5e-4)
        center_optimizer = optim.SGD(criterion_center.parameters(), lr=0.5)

        # original from the paper

        def lr_lambda1(epoch):
            if epoch < 10:
                return (epoch + 1) / 10
            if epoch < 40:
                return 1
            elif epoch < 70:
                return 0.1
            else:
                return 0.01

        def lr_lambda2(epoch):
            if epoch < 150:
                return 1.0
            elif epoch < 180:
                return 0.1
            else:
                return 0.01

        if lr_function == 1:
            lr_lambda = lr_lambda1
        else:
            lr_lambda = lr_lambda2

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        train_loss_hist = [[], [], [], []]
        val_loss_hist = [[], [], [], []]
        rank_hist = [[], [], []]

        model.train()
        for epoch in range(num_epochs):
            running_loss = running_id = running_triplet = running_center = 0.0

            for i, (images, labels, _) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                embeddings, logits = model(images)

                loss_triplet = criterion_triplet(embeddings, labels)
                loss_id = criterion_id(logits, labels)
                loss_center = criterion_center(embeddings, labels)

                running_id += loss_id.item()
                running_triplet += loss_triplet.item()
                total_loss = loss_id + loss_triplet

                if bot_level_train >= 2:
                    total_loss += center_loss_weight * loss_center
                    running_center += (center_loss_weight * loss_center).item()

                optimizer.zero_grad()
                center_optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                if bot_level_train >= 3:
                    for param in criterion_center.parameters():
                        param.grad.data *= 1.0 / center_loss_weight
                    center_optimizer.step()

                running_loss += total_loss.item()

                if (i + 1) % 10 == 0:
                    print(
                        f"  Fold [{fold + 1}] Epoch [{epoch + 1}/{num_epochs}] "
                        f"Step [{i + 1}/{len(train_loader)}] "
                        f"Loss: {total_loss.item():.4f} "
                        f"(ID: {loss_id.item():.2f}, Trp: {loss_triplet.item():.2f})"
                    )

            n_train = len(train_loader)
            train_loss_hist[0].append(running_loss / n_train)
            train_loss_hist[1].append(running_id / n_train)
            train_loss_hist[2].append(running_triplet / n_train)
            train_loss_hist[3].append(running_center / n_train)

            scheduler.step()

            avg_val, val_id, val_trip, val_cen = validate_one_epoch(
                model,
                val_loader,
                criterion_triplet,
                criterion_id,
                criterion_center,
                center_loss_weight,
                bot_level_train,
                device,
            )

            val_loss_hist[0].append(avg_val)
            val_loss_hist[1].append(val_id)
            val_loss_hist[2].append(val_trip)
            val_loss_hist[3].append(val_cen)

            r1, r5, map = calc_one_epoch_r1_map(model, val_loader, device)
            rank_hist[0].append(r1)
            rank_hist[1].append(r5)
            rank_hist[2].append(map)
            model.train()

            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"  >> Fold {fold + 1} | Epoch {epoch + 1}/{num_epochs} | "
                f"LR: {current_lr:.6f} | "
                f"Train Loss: {running_loss / n_train:.4f} | "
                f"Val Loss: {avg_val:.4f} | "
                f"Val R1 Rank {r1:.4f} | "
                f"Val mAP {map:.4f} | "
            )

            torch.save(
                {
                    "epoch": epoch,
                    "fold": fold,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(
                    CHECKPOINT_PATH, f"checkpoint_fold{fold}_epoch{epoch}.pth"
                ),
            )

        fold_save_name = save_name.replace(".pth", f"_fold{fold}.pth")
        torch.save(model.state_dict(), fold_save_name)
        print(f"  [Saved] {fold_save_name}")

        if plot_loss:
            plot_single_fold(
                train_loss_hist,
                val_loss_hist,
                num_epochs,
                title=f"{model_name}_fold{fold}_train_vs_val",
            )

        all_fold_train_losses.append(train_loss_hist)
        all_fold_val_losses.append(val_loss_hist)
        all_fold_ranks.append(rank_hist)

    if plot_loss:
        plot_fold_summary(
            all_fold_train_losses,
            all_fold_val_losses,
            all_fold_ranks,
            num_epochs,
            model_name,
        )

    print("\nTraining Finished (all folds)")
    return all_fold_train_losses, all_fold_val_losses


def train_grid_search():
    for i in range(4):
        for j in range(3):
            train_losses, val_losses = train(
                refresh_data=False,
                plot_loss=True,
                model_name=MOBILENETV3_NAME,
                bot_level_model=i,
                bot_level_train=j,
                save_name=f"{MOBILENETV3_NAME}_botm{i}_bott{j}.pth",
            )


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process model settings.")
    parser.add_argument(
        "-red", "--refresh_data", action="store_true", help="Redownload/ReUnzip data"
    )
    parser.add_argument(
        "-pltls",
        "--plot_loss",
        action="store_true",
        help="Plot the loss of training data",
    )
    parser.add_argument(
        "-pltv",
        "--plot_validation",
        action="store_true",
        help="Plot the validation results from training",
    )
    parser.add_argument(
        "-tgs",
        "--train_grid_search",
        action="store_true",
        help="Do training with all BoT level combinations",
    )
    parser.add_argument(
        "-kf",
        "--kfold",
        action="store_true",
        help="Use 5-fold cross-validation instead of standard training",
    )
    parser.add_argument(
        "-bm",
        "--bot_model",
        type=int,
        default=1,
        help="Set BoT level for the model itself",
    )
    parser.add_argument(
        "-bt",
        "--bot_train",
        type=int,
        default=1,
        help="Set BoT level for the training function",
    )
    parser.add_argument(
        "-mbnet",
        "--mobilenet",
        action="store_const",
        const=MOBILENETV3_NAME,
        dest="model",
        help="Use MobileNetV3",
    )

    parser.add_argument(
        "-ne",
        "--number_epochs",
        type=int,
        default=120,
        help="Number of epochs to train the model on",
    )

    parser.add_argument(
        "-lrf",
        "--learning_rate_func",
        type=int,
        default=1,
        help="Use the suggewsted learning function",
    )

    parser.set_defaults(model=RESNET18_NAME)
    return parser.parse_args()


if __name__ == "__main__":
    torch.manual_seed(42)
    args = parse_arguments()

    if not os.path.isdir(WEIGHTS_PATH):
        os.makedirs(WEIGHTS_PATH)

    shared_kwargs = dict(
        refresh_data=args.refresh_data,
        plot_loss=args.plot_loss,
        model_name=args.model,
        bot_level_model=args.bot_model,
        bot_level_train=args.bot_train,
        lr_function=args.learning_rate_func,
        num_epochs=args.number_epochs,
        save_name=f"{WEIGHTS_PATH}{args.model}_bm{args.bot_model}_bt{args.bot_train}_ne{args.number_epochs}_lr{args.learning_rate_func}_weights.pth",
    )

    if args.train_grid_search:
        train_grid_search()
    elif args.kfold:
        train_kfold(**shared_kwargs)
    else:
        loss_hist = train(**shared_kwargs)
        graph_loss(
            loss_hist,
            epochs=len(loss_hist[0]),
            loss_names=["Avg Loss", "ID Loss", "Triplet Loss", "Center Loss"],
            title=f"{args.model}_loss",
        )
