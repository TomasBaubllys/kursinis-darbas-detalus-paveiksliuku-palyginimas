# Author Tomas Baublys

import random

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.utils
from torch.utils.data import DataLoader

from dataset import MarketSiameseDataset
from utils import imgshow

if __name__ == "__main__":
    transformation = transforms.Compose(
        [transforms.Resize((100, 100)), transforms.ToTensor()]
    )

    siamese_dataset = MarketSiameseDataset("", transformation)

    dataloader = DataLoader(siamese_dataset, shuffle=True, num_workers=2, batch_size=8)

    example_batch = next(iter(dataloader))

    concatanated = torch.cat((example_batch[0], example_batch[1]), 0)
    imgshow(torchvision.utils.make_grid(concatanated))
    print(example_batch[2].numpy().reshape(-1))
