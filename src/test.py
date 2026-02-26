import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from dataset import MarketSiameseDataset
from siamese_network import Siamese_Network

DEFAULT_DATA_PATH = "../data/Market-1501-v15.09.15/"


class MarketEvalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = []  # list of (img_path, pid, camid)

        full_dir = os.path.join(os.getcwd(), root_dir)
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
            emb = nn.functional.normalize(emb, p=2, dim=1)
            all_features.append(emb.cpu().numpy())
            all_pids.extend(pids)
            all_camids.extend(camids)

    return np.vstack(all_features), np.array(all_pids), np.array(all_camids)


def compute_metrics(dist_matrix, q_pids, g_pids, q_camids, g_camids):
    """
    Proper Market-1501 evaluation:
    For each query, remove gallery images with same pid AND same camera
    (junk images) before ranking.
    """
    num_queries = len(q_pids)
    rank1_correct = 0
    rank5_correct = 0
    all_ap = []

    for i in range(num_queries):
        q_pid = q_pids[i]
        q_camid = q_camids[i]

        # Find valid gallery indices: exclude same pid + same camera
        valid_mask = ~((g_pids == q_pid) & (g_camids == q_camid))

        valid_dist = dist_matrix[i][valid_mask]
        valid_pids = g_pids[valid_mask]

        rank_indices = np.argsort(valid_dist)
        ranked_pids = valid_pids[rank_indices]

        matches = (ranked_pids == q_pid).astype(int)

        if matches.sum() == 0:
            continue

        # Rank-1
        if ranked_pids[0] == q_pid:
            rank1_correct += 1

        # Rank-5
        if np.any(ranked_pids[:5] == q_pid):
            rank5_correct += 1

        # Average Precision
        precision_at_k = np.cumsum(matches) / (np.arange(len(matches)) + 1)
        ap = np.sum(precision_at_k * matches) / matches.sum()
        all_ap.append(ap)

    return (
        rank1_correct / num_queries * 100,
        rank5_correct / num_queries * 100,
        np.mean(all_ap) * 100,
    )


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

    train_path = os.path.join(DEFAULT_DATA_PATH, "bounding_box_train")
    train_ds = MarketSiameseDataset(train_path)
    num_classes = train_ds.num_ids

    print(f"Initializing model with {num_classes} classes...")
    model = Siamese_Network(num_classes=num_classes).to(device)

    checkpoint_name = "siamese_resnet18_bot.pth"
    if os.path.exists(checkpoint_name):
        model.load_state_dict(torch.load(checkpoint_name, map_location=device))
        print(f"Loaded {checkpoint_name}")
    else:
        print("Warning: Checkpoint not found, using random weights.")

    model.eval()

    gallery_ds = MarketEvalDataset(
        os.path.join(DEFAULT_DATA_PATH, "bounding_box_test"), transform=transform
    )
    query_ds = MarketEvalDataset(
        os.path.join(DEFAULT_DATA_PATH, "query"), transform=transform
    )

    gallery_loader = DataLoader(gallery_ds, batch_size=64, shuffle=False, num_workers=4)
    query_loader = DataLoader(query_ds, batch_size=64, shuffle=False, num_workers=4)

    print(f"Query images: {len(query_ds)}, Gallery images: {len(gallery_ds)}")
    print("Extracting features...")

    q_feat, q_pids, q_camids = extract_features(model, query_loader, device)
    g_feat, g_pids, g_camids = extract_features(model, gallery_loader, device)

    print("Computing distance matrix (Cosine) and metrics...")
    dist_matrix = cdist(q_feat, g_feat, metric="cosine")

    r1, r5, mAP = compute_metrics(dist_matrix, q_pids, g_pids, q_camids, g_camids)

    print("\n" + "=" * 30)
    print("EVALUATION (COSINE DISTANCE)")
    print("=" * 30)
    print(f"Query:   {len(q_pids)} images")
    print(f"Gallery: {len(g_pids)} images")
    print(f"Rank-1:  {r1:.2f}%")
    print(f"Rank-5:  {r5:.2f}%")
    print(f"mAP:     {mAP:.2f}%")
    print("=" * 30)


if __name__ == "__main__":
    evaluate()
