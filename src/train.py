import os
import sys

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
    if len(sys.argv) > 1:
        args = sys.argv[1:]
        for arg in args:
            if arg in ("-red", "--refresh_data"):
                download_data.setup_market1501()

    # make a checkpoint dir
    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html
    transformations = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

    num_epochs = 60

    siamese_net.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # forward pass
            embeddings = siamese_net(images)

            loss = criterion(embeddings, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}"
                )

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1} done. Current Learning Rate: {current_lr}, Average Loss: {running_loss / len(dataloader):.4f}"
        )

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
    print("Training Finished")


if __name__ == "__main__":
    train()
