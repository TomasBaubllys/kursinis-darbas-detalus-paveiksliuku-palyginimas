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
            camid = parts[1]
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
            emb = model(imgs)  # This returns bn_feat (fi) in eval mode
            # L2 Normalize for Cosine Similarity / Re-ranking
            emb = nn.functional.normalize(emb, p=2, dim=1)
            all_features.append(emb.cpu())
            all_pids.extend(pids)
            all_camids.extend(camids)

    return torch.cat(all_features, dim=0), np.array(all_pids), np.array(all_camids)


def re_ranking(probFea, galFea, k1=20, k2=6, lambda_value=0.3):
    """
    Re-ranking Person Re-identification with k-reciprocal Encoding
    Optimized for GPU/CPU tensors
    """
    query_num = probFea.size(0)
    all_num = query_num + galFea.size(0)
    feat = torch.cat([probFea, galFea])

    # Compute Euclidean distance matrix
    distmat = (
        torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num)
        + torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num).t()
    )
    distmat.addmm_(1, -2, feat, feat.t())
    original_dist = distmat.cpu().numpy()
    del feat

    gallery_num = original_dist.shape[0]
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float16)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    print("Starting re-ranking process...")
    for i in range(all_num):
        forward_k_neigh_index = initial_rank[i, : k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, : k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index

        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[
                candidate, : int(np.around(k1 / 2)) + 1
            ]
            candidate_backward_k_neigh_index = initial_rank[
                candidate_forward_k_neigh_index, : int(np.around(k1 / 2)) + 1
            ]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(
                np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)
            ) > 2 / 3 * len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(
                    k_reciprocal_expansion_index, candidate_k_reciprocal_index
                )

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)

    original_dist = original_dist[:query_num,]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe

    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)
    for i in range(query_num):
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float16)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(
                V[i, indNonZero[j]], V[indImages[j], indNonZero[j]]
            )
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    return final_dist[:query_num, query_num:]


def compute_metrics(dist_matrix, q_pids, g_pids, q_camids, g_camids):
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
            continue

        if ranked_pids[0] == q_pid:
            rank1_correct += 1
        if np.any(ranked_pids[:5] == q_pid):
            rank5_correct += 1

        # Average Precision
        precision_at_k = np.cumsum(matches) / (np.arange(len(matches)) + 1)
        ap = np.sum(precision_at_k * matches) / matches.sum()
        all_ap.append(ap)

    # FIXED: Moved outside the for loop
    return (
        rank1_correct / num_queries * 100,
        rank5_correct / num_queries * 100,
        np.mean(all_ap) * 100 if all_ap else 0,
    )


def evaluate(use_reranking=True):
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

    model = ResNet18_BoT(num_classes=num_classes).to(device)

    checkpoint_name = "resnet18_bot.pth"
    if os.path.exists(checkpoint_name):
        model.load_state_dict(torch.load(checkpoint_name, map_location=device))
        print(f"Loaded {checkpoint_name}")

    query_ds = MarketEvalDataset(
        os.path.join(DEFAULT_DATA_PATH, "query"), transform=transform
    )
    gallery_ds = MarketEvalDataset(
        os.path.join(DEFAULT_DATA_PATH, "bounding_box_test"), transform=transform
    )

    query_loader = DataLoader(query_ds, batch_size=64, shuffle=False, num_workers=4)
    gallery_loader = DataLoader(gallery_ds, batch_size=64, shuffle=False, num_workers=4)

    q_feat, q_pids, q_camids = extract_features(model, query_loader, device)
    g_feat, g_pids, g_camids = extract_features(model, gallery_loader, device)

    if use_reranking:
        print("Computing distance matrix with k-reciprocal re-ranking...")
        dist_matrix = re_ranking(q_feat, g_feat)
    else:
        print("Computing distance matrix with standard cosine distance...")
        # Convert to numpy for cdist if not re-ranking
        dist_matrix = cdist(q_feat.numpy(), g_feat.numpy(), metric="cosine")

    r1, r5, mAP = compute_metrics(dist_matrix, q_pids, g_pids, q_camids, g_camids)

    print(f"\nRESULTS {'(WITH RE-RANKING)' if use_reranking else '(BASELINE)'}")
    print(f"Rank-1: {r1:.2f}% | Rank-5: {r5:.2f}% | mAP: {mAP:.2f}%")


if __name__ == "__main__":
    evaluate(use_reranking=True)
