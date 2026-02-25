import os
import sys
from tabnanny import verbose

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import download_data
from dataset import DEFAULT_DATA_PATH, MarketSiameseDataset
from pksampler import PKSampler
from siamese_network import Siamese_Network
from triplet_loss import Batch_Hard_Triplet_Loss

CHECKPOINT_PATH = "../checkpoint"


def train():
    # check if user wanted to redownload data (re unzip)
    plot_loss = False

    if len(sys.argv) > 1:
        args = sys.argv[1:]
        for arg in args:
            if arg in ("-red", "--refresh_data"):
                download_data.setup_market1501()
            if arg in ("-plst", "--plot_loss"):
                plot_loss = True

                # make a checkpoint dir
    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html
    # transformations = transforms.Compose(
    # [
    # transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
    # transforms.CenterCrop(224),
    # transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ]
    # )
    #

    transformations = transforms.Compose(
        [
            transforms.Resize(
                (256, 128), interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.5, value="random"),
        ]
    )

    market_dataset = MarketSiameseDataset(DEFAULT_DATA_PATH, transform=transformations)
    p_val = 16  # how many total people per batch
    k_val = 4  # this is how many of the same person images per batch
    market_sampler = PKSampler(market_dataset, p=p_val, k=k_val)

    dataloader = DataLoader(
        market_dataset, sampler=market_sampler, num_workers=2, batch_size=p_val * k_val
    )

    siamese_net = Siamese_Network().to(device)
    criterion = Batch_Hard_Triplet_Loss(margin=0.3).to(device)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, siamese_net.parameters()), lr=0.001
    )

    # train 40 epochs at the starting 0.001 and the next 20 at 0.0001
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5, min_lr=1e-6
    )

    num_epochs = 90

    loss_hist = []

    siamese_net.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            embeddings = siamese_net(images)

            embeddings = nn.functional.normalize(embeddings, p=2, dim=1)

            loss = criterion(embeddings, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}"
                )

        avg_loss = running_loss / len(dataloader)
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1} done. Current Learning Rate: {current_lr}, Average Loss: {running_loss / len(dataloader):.4f}"
        )

        loss_hist.append([epoch, avg_loss])

        # save everything
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": siamese_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

        torch.save(
            checkpoint,
            os.path.join(os.getcwd(), CHECKPOINT_PATH, f"checkpoint_{epoch=}.pth"),
        )

        # save the final state
    torch.save(siamese_net.state_dict(), "siamese_resnet18_batchhard.pth")

    if plot_loss:
        plt.plot(np.array(loss_hist))
        plt.savefig("loss_hist.jpg")
        plt.show()

    print("Training Finished")


"""
CHECKPOINT_PATH = "../checkpoint"


def train():
    # --- CLI Arguments & Setup ---
    plot_loss = False
    if len(sys.argv) > 1:
        args = sys.argv[1:]
        for arg in args:
            if arg in ("-red", "--refresh_data"):
                download_data.setup_market1501()
            if arg in ("-plst", "--plot_loss"):
                plot_loss = True

    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Data Prep ---
    transformations = transforms.Compose(
        [
            transforms.Resize(
                (256, 128), interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.5, value="random"),
        ]
    )

    market_dataset = MarketSiameseDataset(DEFAULT_DATA_PATH, transform=transformations)

    # Label Remapping Fix
    raw_ids = sorted(list(market_dataset.id_to_images.keys()))
    id_map = {raw_id: i for i, raw_id in enumerate(raw_ids)}
    new_id_to_images = {
        id_map[raw_id]: paths for raw_id, paths in market_dataset.id_to_images.items()
    }
    market_dataset.id_to_images = new_id_to_images
    market_dataset.person_ids = list(new_id_to_images.keys())

    num_classes = len(market_dataset.person_ids)
    print(f"Mapped {num_classes} identities. Range: [0, {num_classes - 1}]")

    # P=16, K=4 or K=8 is standard for Triplet Loss
    p_val, k_val = 16, 8
    market_sampler = PKSampler(market_dataset, p=p_val, k=k_val)
    dataloader = DataLoader(
        market_dataset, sampler=market_sampler, num_workers=4, batch_size=p_val * k_val
    )

    # --- Model & Loss ---
    siamese_net = Siamese_Network(num_classes=num_classes).to(device)
    triplet_criterion = Batch_Hard_Triplet_Loss(margin=0.3).to(device)
    id_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Optimizer
    optimizer = optim.Adam(siamese_net.parameters(), lr=0.00035, weight_decay=5e-4)

    # Scheduler: Warmup for 10 epochs, then Step decay at 40 and 70
    # For simplicity, we use MultiStepLR. To reach 85%, use a custom warmup scheduler.
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[40, 70], gamma=0.1
    )

    num_epochs = 90
    loss_hist = []

    # --- Training Loop ---
    siamese_net.train()
    print("Starting Training...")

    for epoch in range(num_epochs):
        running_loss = 0.0
        running_id_loss = 0.0
        running_tri_loss = 0.0

        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            embeddings, cls_scores = siamese_net(images)

            # Normalize for Triplet Loss calculation
            norm_embeddings = nn.functional.normalize(embeddings, p=2, dim=1)

            loss_tri = triplet_criterion(norm_embeddings, labels)
            loss_id = id_criterion(cls_scores, labels)

            total_loss = loss_tri + loss_id

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            running_id_loss += loss_id.item()
            running_tri_loss += loss_tri.item()

            if (i + 1) % 20 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], "
                    f"Total Loss: {total_loss.item():.4f} (ID: {loss_id.item():.4f}, Tri: {loss_tri.item():.4f})"
                )

        avg_loss = running_loss / len(dataloader)
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"--- Epoch {epoch + 1} Done | Avg Loss: {avg_loss:.4f} | LR: {current_lr} ---"
        )

        loss_hist.append(avg_loss)

        # Save Checkpoint every 5 epochs or last epoch
        if (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": siamese_net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            }
            torch.save(
                checkpoint,
                os.path.join(CHECKPOINT_PATH, f"checkpoint_epoch_{epoch + 1}.pth"),
            )

    # Save Final Model
    torch.save(siamese_net.state_dict(), "siamese_resnet18_final.pth")

    if plot_loss:
        plt.figure(figsize=(10, 5))
        plt.plot(loss_hist, label="Total Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.legend()
        plt.savefig("loss_hist.jpg")
        plt.show()

    print("Training Finished Successfully.")

"""
if __name__ == "__main__":
    train()
