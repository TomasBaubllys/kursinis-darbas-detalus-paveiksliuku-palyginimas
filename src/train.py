import os
import sys
import argparse

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
from sklearn.model_selection import GroupKFold

RESNET18_NAME = "resnet18"
MOBILENETV3_NAME = "mobilenetv3"
CHECKPOINT_PATH = "../checkpoint"


def graph_loss(loss_hist, epochs, loss_names=["1", "2"], title=""):
    x = np.arange(epochs)
    plt.figure()
    for i, name in enumerate(loss_names):
        data = [val.detach().cpu().item() if hasattr(val, 'cpu') else val for val in loss_hist[i]]
        plt.plot(x, data, label=name)

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{title}")
    plt.legend()
    plt.savefig(f"{title}.jpg")
    plt.close()


def train(
    refresh_data=False,
    plot_loss=False,
    save_name="checkpoint.pth",
    model_name=RESNET18_NAME,
    bot_level_model=0,
    bot_level_train=0,
):
    if refresh_data:
        download_data.setup_market1501()

    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if bot_level_train >= 1:
        transformations = transforms.Compose(
            [
                transforms.Resize(
                    (256, 128), interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Pad(10),
                transforms.RandomCrop((256, 128)),
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
        transformations = transforms.Compose(
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
        model = ResNet18_BoT(num_classes=num_classes).to(device)

    criterion_triplet = Batch_Hard_Triplet_Loss(margin=0.3).to(device)
    criterion_id = Cross_Entropy_Label_Smooth(num_classes=num_classes).to(device)

    if model_name == MOBILENETV3_NAME:
        criterion_center = Center_Loss(
            num_classes=num_classes, feat_dim=model.feat_dim
        ).to(device)
    elif model_name == RESNET18_NAME:
        criterion_center = Center_Loss(num_classes=num_classes).to(device)

    center_loss_weight = 0.0005

    optimizer = optim.Adam(model.parameters(), lr=3.5e-4, weight_decay=5e-4)
    center_optimizer = optim.SGD(criterion_center.parameters(), lr=0.5)

    def lr_lambda(epoch):
        if epoch < 10:
            return (epoch + 1) / 10
        elif epoch < 40:
            return 1
        elif epoch < 70:
            return 0.1
        else:
            return 0.01

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    num_epochs = 120
    loss_hist = [[], [], [], []]

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_id_loss = 0.0
        running_triplet_loss = 0.0
        running_center_loss = 0.0
        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)

            embeddings, logits = model(images)

            # calculate losses
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

            if bot_level_train >= 2:
                for param in criterion_center.parameters():
                    param.grad.data *= 1.0 / 0.0005

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
    return loss_hist


def train_grid_search():
    for i in range(4):
        for j in range(3):
            loss = train(
                refresh_data=False,
                model_name=MOBILENETV3_NAME,
                bot_level_model=i,
                bot_level_train=j,
                save_name=f"{MOBILENETV3_NAME}_botm{i}_bott{j}.pth",
            )
            graph_loss(loss, len(loss), f"{MOBILENETV3_NAME}_botm{i}_bott{j}" )

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process model settings.")

    parser.add_argument("-red", "--refresh_data", action="store_true", help="Redownload/ReUnzip data")
    parser.add_argument("-pltls", "--plot_loss", action="store_true", help="Plot the loss of training data")
    parser.add_argument("-pltv", "--plot_validation", action="store_true", help="Plot the validation results from training")
    parser.add_argument("-tgs", "--train_grid_search", action="store_true", help="Do training with all the possible combinations of BoT levels")


    parser.add_argument("-bm", "--bot_model", type=int, default=1, help="Set BoT level for the model itself")

    parser.add_argument("-bt", "--bot_train", type=int, default=1, help="Set BoT level for the training function")

    parser.add_argument("-mbnet", "--mobilenet", action="store_const",
                        const="mobilenetv3", dest="model", help="Use MobileNetV3")

    return parser.parse_args()


if __name__ == "__main__":
    #r_data = False
    #plot_lst = False
    #botm = 0
    #bott = 0
    #model = RESNET18_NAME
    args = parse_arguments()

    #if len(sys.argv) > 1:
    #    args = sys.argv[1:]

    #    for arg in args:
    #        if arg in ("-red", "--refresh_data"):
    #            r_data = True
    #        if arg in ("-plst", "--plot_loss"):
    #            plot_lst = True
    #        if arg.startswith(("-bm", "--bot_model")):
    #            botm = int(arg.replace("--bot_model", "").replace("-bm", ""))loss_names
    #        if arg.startswith(("-bt", "--bot_train")):
    #            bott = int(arg.replace("--bot_train", "").replace("-bt", ""))
    #        if arg in ("-mbnet", "--mobilenet"):
    #            model = MOBILENETV3_NAME
    #        if arg in ("-tgs", "--train_grid_search"):
    #            train_grid_search()
    #            exit()loss_names
    loss = train(
        refresh_data=args.refresh_data,
        plot_loss=args.plot_loss,
        model_name=args.model,
        bot_level_model=args.bot_model,
        bot_level_train=args.bot_train,
        save_name=f"{args.model}_weights.pth",
    )

    graph_loss(loss, epochs=len(loss[0]), loss_names=["Avg Loss", "ID Loss", "Triplet Loss", "Center Loss"], title=f"{MOBILENETV3_NAME}_botm{0}_bott{0}" )
