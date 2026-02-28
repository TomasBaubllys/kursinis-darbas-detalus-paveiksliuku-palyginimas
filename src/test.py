import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Assuming these are in your local directory
from dataset import MarketSiameseDataset
from resnet18_bot import ResNet18_BoT

DEFAULT_DATA_PATH = "../data/Market-1501-v15.09.15/"


class MarketEvalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = []  # list of (img_path, pid, camid)

        full_dir = os.path.join(os.getcwd(), root_dir)
        if not os.path.exists(full_dir):
            raise FileNotFoundError(f"Directory {full_dir} not found.")

        for f in sorted(os.listdir(full_dir)):
            if not f.endswith(".jpg"):
                continue
            parts = f.split("_")
            pid = parts[0]
            camid = parts[1]  # e.g. "c1"
            if pid in ["-1", "0000"]:
                continue
            self.samples.append((os.path.join(full_dir, f), pid, camid))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, pid, camid = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, pid, camid


def extract_features(model, dataloader, device):
    model.eval()
    all_features = []
    all_pids = []
    all_camids = []

    with torch.no_grad():
        for imgs, pids, camids in dataloader:
            imgs = imgs.to(device)
            emb = model(imgs)
            # L2 Normalize for Cosine Similarity
            emb = nn.functional.normalize(emb, p=2, dim=1)
            all_features.append(emb.cpu().numpy())
            all_pids.extend(pids)
            all_camids.extend(camids)

    return np.vstack(all_features), np.array(all_pids), np.array(all_camids)


def compute_metrics(dist_matrix, q_pids, g_pids, q_camids, g_camids):
    queries_with_matches = 0
    num_queries = len(q_pids)
    rank1_correct = 0
    rank5_correct = 0
    all_ap = []

    for i in range(num_queries):
        q_pid = q_pids[i]
        q_camid = q_camids[i]

        # Market-1501 Rule: Exclude gallery images with same PID AND same Camera (Junk)
        valid_mask = ~((g_pids == q_pid) & (g_camids == q_camid))
        valid_dist = dist_matrix[i][valid_mask]
        valid_pids = g_pids[valid_mask]

        rank_indices = np.argsort(valid_dist)
        ranked_pids = valid_pids[rank_indices]

        matches = (ranked_pids == q_pid).astype(int)
        if matches.sum() == 0:
            queries_with_matches += 1
            continue

        if ranked_pids[0] == q_pid:
            rank1_correct += 1
        if np.any(ranked_pids[:5] == q_pid):
            rank5_correct += 1

        # Average Precision
        precision_at_k = np.cumsum(matches) / (np.arange(len(matches)) + 1)
        ap = np.sum(precision_at_k * matches) / matches.sum()
        all_ap.append(ap)

        r1 = (
            (rank1_correct / queries_with_matches * 100)
            if queries_with_matches > 0
            else 0
        )
        r5 = (
            (rank5_correct / queries_with_matches * 100)
            if queries_with_matches > 0
            else 0
        )
        mAP = (np.mean(all_ap) * 100) if all_ap else 0

        return r1, r5, mAP

        # return (
        # rank1_correct / num_queries * 100,
        # rank5_correct / num_queries * 100,
        # np.mean(all_ap) * 100 if all_ap else 0,
        # )


def visualize_results(
    q_ds, g_ds, dist_matrix, q_pids, g_pids, q_camids, g_camids, num_samples=16
):
    """
    Displays query images and their top-1 gallery match with color-coded borders.
    """
    print(f"Generating visualization for {num_samples} random queries...")
    plt.figure(figsize=(16, 12))
    indices = np.random.choice(len(q_pids), num_samples, replace=False)

    # Values for un-normalizing images
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for i, q_idx in enumerate(indices):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # Filter junk gallery images for this specific query
        valid_mask = ~((g_pids == q_pid) & (g_camids == q_camid))
        valid_indices = np.where(valid_mask)[0]
        valid_distances = dist_matrix[q_idx][valid_mask]

        # Get Rank-1 index
        top1_local_idx = np.argmin(valid_distances)
        top1_global_idx = valid_indices[top1_local_idx]

        pred_pid = g_pids[top1_global_idx]
        is_correct = pred_pid == q_pid
        color = "green" if is_correct else "red"

        # --- Helper to process tensor to image ---
        def denormalize(tensor):
            img = tensor.permute(1, 2, 0).numpy()
            img = img * std + mean
            return np.clip(img, 0, 1)

        # Plot Query
        ax_q = plt.subplot(4, 8, 2 * i + 1)
        plt.imshow(denormalize(q_ds[q_idx][0]))
        plt.title(f"Q: {q_pid}", fontsize=9)
        plt.axis("off")

        # Plot Match
        ax_m = plt.subplot(4, 8, 2 * i + 2)
        plt.imshow(denormalize(g_ds[top1_global_idx][0]))
        plt.title(f"R1: {pred_pid}", color=color, fontsize=9, fontweight="bold")

        # Add colored border
        for spine in ax_m.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
            spine.set_visible(True)
        ax_m.set_xticks([])
        ax_m.set_yticks([])

    plt.suptitle("Person Re-ID Results: Green=Correct, Red=Incorrect", fontsize=16)
    plt.tight_layout()
    plt.savefig("res.jpg")
    plt.show()


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [
            transforms.Resize(
                (256, 128), interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Get number of classes from training set to initialize model architecture
    train_path = os.path.join(DEFAULT_DATA_PATH, "bounding_box_train")
    train_ds = MarketSiameseDataset(train_path)
    num_classes = train_ds.num_ids

    print(f"Initializing model with {num_classes} classes...")
    model = ResNet18_BoT(num_classes=num_classes).to(device)

    checkpoint_name = "siamese_resnet18_bot.pth"
    if os.path.exists(checkpoint_name):
        model.load_state_dict(torch.load(checkpoint_name, map_location=device))
        print(f"Loaded {checkpoint_name}")
    else:
        print("Warning: Checkpoint not found, using random weights.")

    gallery_ds = MarketEvalDataset(
        os.path.join(DEFAULT_DATA_PATH, "bounding_box_test"), transform=transform
    )
    query_ds = MarketEvalDataset(
        os.path.join(DEFAULT_DATA_PATH, "query"), transform=transform
    )

    gallery_loader = DataLoader(gallery_ds, batch_size=64, shuffle=False, num_workers=4)
    query_loader = DataLoader(query_ds, batch_size=64, shuffle=False, num_workers=4)

    print(f"Query images: {len(query_ds)}, Gallery images: {len(gallery_ds)}")

    q_feat, q_pids, q_camids = extract_features(model, query_loader, device)
    g_feat, g_pids, g_camids = extract_features(model, gallery_loader, device)

    print("Computing distance matrix...")
    dist_matrix = cdist(q_feat, g_feat, metric="cosine")

    r1, r5, mAP = compute_metrics(dist_matrix, q_pids, g_pids, q_camids, g_camids)

    print("\n" + "=" * 30)
    print("EVALUATION RESULTS")
    print("=" * 30)
    print(f"Rank-1:  {r1:.2f}%")
    print(f"Rank-5:  {r5:.2f}%")
    print(f"mAP:     {mAP:.2f}%")
    print("=" * 30 + "\n")

    # Run the visualization
    visualize_results(
        query_ds, gallery_ds, dist_matrix, q_pids, g_pids, q_camids, g_camids
    )


if __name__ == "__main__":
    evaluate()
