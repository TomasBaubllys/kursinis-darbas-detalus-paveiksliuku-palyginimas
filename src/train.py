import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import download_data
from center_loss import Center_Loss
from cross_entropy_label_smooth import Cross_Entropy_Label_Smooth
from dataset import DEFAULT_DATA_PATH, Market_Train_Dataset
from models import MobileNetV3_BoT, ResNet18_BoT
from pksampler import PKSampler
from triplet_loss import Batch_Hard_Triplet_Loss

CHECKPOINT_PATH = "../checkpoint"


def train(
    refresh_data=False,
    plot_loss=False,
    save_name="checkpoint.pth",
    model_name="resnet18",
    bot=False,
):
    if refresh_data:
        download_data.setup_market1501()

    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transformations = transforms.Compose(
        [
            transforms.Resize(
                (256, 128), interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Pad(10),
            transforms.RandomCrop((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(
                p=0.5, scale=(0.02, 0.4), ratio=(0.3, 3.33), value="random"
            ),
        ]
    )

    market_dataset = Market_Train_Dataset(DEFAULT_DATA_PATH, transform=transformations)
    num_classes = market_dataset.num_ids

    p_val, k_val = 16, 4
    market_sampler = PKSampler(market_dataset, p=p_val, k=k_val)
    dataloader = DataLoader(
        market_dataset, sampler=market_sampler, num_workers=4, batch_size=p_val * k_val
    )

    if model_name == "mobilenetv3":
        model = MobileNetV3_BoT(num_classes=num_classes, bot=bot).to(device)
    elif model_name == "resnet18":
        model = ResNet18_BoT(num_classes=num_classes).to(device)

    criterion_triplet = Batch_Hard_Triplet_Loss(margin=0.3).to(device)
    criterion_id = Cross_Entropy_Label_Smooth(num_classes=num_classes).to(device)

    if model_name == "mobilenetv3":
        criterion_center = Center_Loss(
            num_classes=num_classes, feat_dim=model.feat_dim
        ).to(device)
    elif model_name == "resnet18":
        criterion_center = Center_Loss(num_classes=num_classes).to(device)

    center_loss_weight = 0.0005

    optimizer = optim.Adam(model.parameters(), lr=3.5e-4, weight_decay=5e-4)
    center_optimizer = optim.SGD(criterion_center.parameters(), lr=0.5)

    def lr_lambda(epoch):
        if epoch < 10:
            return (epoch + 1) / 10  # Linear warmup
        elif epoch < 40:
            return 1
        elif epoch < 70:
            return 0.1
        else:
            return 0.01

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    num_epochs = 120
    loss_hist = []

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)

            embeddings, logits = model(images)

            loss_triplet = criterion_triplet(embeddings, labels)
            loss_id = criterion_id(logits, labels)
            loss_center = criterion_center(embeddings, labels)
            total_loss = loss_triplet + loss_id + center_loss_weight * loss_center

            optimizer.zero_grad()
            center_optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            if (i + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {total_loss.item():.4f} (ID: {loss_id.item():.2f}, Trp: {loss_triplet.item():.2f})"
                )

            for param in criterion_center.parameters():
                param.grad.data *= 1.0 / 0.0005

            center_optimizer.step()

        scheduler.step()
        avg_loss = running_loss / len(dataloader)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1} complete. LR: {current_lr:.6f}, Avg Loss: {avg_loss:.4f}"
        )

        loss_hist.append([epoch, avg_loss])

        # Checkpoint Preservation
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join(CHECKPOINT_PATH, f"checkpoint_{epoch}.pth"))

    torch.save(model.state_dict(), save_name)
    if plot_loss:
        plt.plot(np.array(loss_hist)[:, 1])
        plt.savefig("loss_hist.jpg")
    print("Training Finished")


if __name__ == "__main__":
    r_data = False
    plot_lst = False
    bot = False
    model = "resnet18"

    if len(sys.argv) > 1:
        args = sys.argv[1:]
        for arg in args:
            if arg in ("-red", "--refresh_data"):
                r_data = True
            if arg in ("-plst", "--plot_loss"):
                plot_lst = True
            if arg in ("-b", "--bot"):
                bot = True
            if arg in ("-mbnet", "--mobilenet"):
                model = "mobilenetv3"
    train(
        refresh_data=r_data,
        plot_loss=plot_lst,
        model_name=model,
        bot=bot,
        save_name=f"{model}_weights.pth",
    )
