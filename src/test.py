"""
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
"""

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
    with torch.no_grad():
        for imgs, pids in dataloader:
            imgs = imgs.to(device)
            # Forward pass
            emb = model(imgs)
            # IMPORTANT: Normalize just like you did in training
            emb = nn.functional.normalize(emb, p=2, dim=1)
            features.append(emb.cpu().numpy())
            labels.append(pids.numpy())
    return np.vstack(features), np.concatenate(labels)


def compute_metrics(dist_matrix, q_pids, g_pids):
    """
    Compute Rank-1, Rank-5, Rank-10, and mAP.

    Args:
        dist_matrix: (num_queries, num_gallery) distance matrix
        q_pids: query person IDs
        g_pids: gallery person IDs

    Returns:
        rank1, rank5, rank10, mAP: evaluation metrics as percentages
    """
    num_queries = len(q_pids)

    rank1_correct = 0
    rank5_correct = 0
    rank10_correct = 0
    all_ap = []

    for i in range(num_queries):
        # Sort gallery images by distance to this query
        rank_indices = np.argsort(dist_matrix[i])

        # Get person IDs in ranked order
        ranked_pids = g_pids[rank_indices]

        # Check if closest match is correct (Rank-1)
        if ranked_pids[0] == q_pids[i]:
            rank1_correct += 1

        # Check if any of top-5 are correct (Rank-5)
        if np.any(ranked_pids[:5] == q_pids[i]):
            rank5_correct += 1

        # Check if any of top-10 are correct (Rank-10)
        if np.any(ranked_pids[:10] == q_pids[i]):
            rank10_correct += 1

        # Compute Average Precision (AP)
        # Mark which gallery images match the query
        matches = (ranked_pids == q_pids[i]).astype(int)

        if np.sum(matches) > 0:
            # Precision at each position
            precision_at_k = np.cumsum(matches) / (np.arange(len(matches)) + 1)
            # Average Precision = sum of (precision * match) / num_matches
            ap = np.sum(precision_at_k * matches) / np.sum(matches)
            all_ap.append(ap)
        else:
            # No positive matches for this query
            all_ap.append(0.0)

    rank1 = rank1_correct / num_queries * 100
    rank5 = rank5_correct / num_queries * 100
    rank10 = rank10_correct / num_queries * 100
    mAP = np.mean(all_ap) * 100

    return rank1, rank5, rank10, mAP


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Use the same transforms as training (no augmentation for testing)
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
    print("Loading model...")
    model = Siamese_Network().to(device)
    model.load_state_dict(
        torch.load("siamese_resnet18_batchhard.pth", map_location=device)
    )
    model.eval()
    print("Model loaded successfully")

    # Setup Gallery and Query DataLoaders
    gallery_path = os.path.join(DEFAULT_DATA_PATH, "bounding_box_test")
    query_path = os.path.join(DEFAULT_DATA_PATH, "query")

    print(f"Loading gallery from: {gallery_path}")
    print(f"Loading query from: {query_path}")

    gallery_loader = DataLoader(
        MarketSiameseDataset(gallery_path, transform=transform), batch_size=64
    )
    query_loader = DataLoader(
        MarketSiameseDataset(query_path, transform=transform), batch_size=64
    )

    print("Extracting features...")
    q_feat, q_pids = extract_features(model, query_loader, device)
    g_feat, g_pids = extract_features(model, gallery_loader, device)

    print(f"Query set: {len(q_feat)} samples")
    print(f"Gallery set: {len(g_feat)} samples")

    # Calculate Euclidean Distance between all Query and Gallery images
    print("Computing distance matrix...")
    dist_matrix = cdist(q_feat, g_feat, metric="euclidean")

    # Compute metrics
    print("Computing metrics...")
    rank1, rank5, rank10, mAP = compute_metrics(dist_matrix, q_pids, g_pids)

    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Rank-1 Accuracy:  {rank1:.2f}%")
    print(f"Rank-5 Accuracy:  {rank5:.2f}%")
    print(f"Rank-10 Accuracy: {rank10:.2f}%")
    print(f"Mean Average Precision (mAP): {mAP:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    evaluate()
