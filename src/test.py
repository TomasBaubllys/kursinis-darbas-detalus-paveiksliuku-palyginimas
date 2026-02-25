import os

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import MarketSiameseDataset
from siamese_network import Siamese_Network

DEFAULT_DATA_PATH = "../data/Market-1501-v15.09.15/"


def extract_features(model, dataloader, device):
    model.eval()
    features = []
    labels = []
    cam_ids = []  # Market-1501 requires camera IDs for mAP

    with torch.no_grad():
        for imgs, pids in dataloader:
            imgs = imgs.to(device)
            # 1. Forward pass
            emb = model(imgs)
            # 2. IMPORTANT: Normalize just like you did in training
            emb = nn.functional.normalize(emb, p=2, dim=1)

            features.append(emb.cpu().numpy())
            labels.append(pids.numpy())

    return np.vstack(features), np.concatenate(labels)


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use the same transforms as training
    transform = transforms.Compose(
        [
            transforms.Resize(
                (256, 128), interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load Model
    model = Siamese_Network().to(device)
    model.load_state_dict(torch.load("siamese_resnet18_batchhard.pth"))
    model.eval()

    # Setup Gallery and Query DataLoaders
    # Note: Use your custom Dataset class but point to test folders
    gallery_path = os.path.join(DEFAULT_DATA_PATH, "bounding_box_test")
    query_path = os.path.join(DEFAULT_DATA_PATH, "query")

    gallery_loader = DataLoader(
        MarketSiameseDataset(gallery_path, transform=transform), batch_size=64
    )
    query_loader = DataLoader(
        MarketSiameseDataset(query_path, transform=transform), batch_size=64
    )

    print("Extracting features...")
    q_feat, q_pids = extract_features(model, query_loader, device)
    g_feat, g_pids = extract_features(model, gallery_loader, device)

    # Calculate Euclidean Distance between all Query and Gallery images
    # Shape will be (num_queries, num_gallery)
    dist_matrix = cdist(q_feat, g_feat, metric="euclidean")

    # Rank-1 Accuracy Calculation
    r1 = 0
    for i in range(len(q_pids)):
        # Sort indices of gallery by distance to this query
        rank_indices = np.argsort(dist_matrix[i])

        # Get the PID of the closest gallery match
        closest_match_pid = g_pids[rank_indices[0]]

        if closest_match_pid == q_pids[i]:
            r1 += 1

    print(f"Rank-1 Accuracy: {r1 / len(q_pids) * 100:.2f}%")


if __name__ == "__main__":
    evaluate()
