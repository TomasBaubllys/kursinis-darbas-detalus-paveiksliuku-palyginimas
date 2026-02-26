import torch
import torch.nn as nn
import torch.nn.functional as F


class Batch_Hard_Triplet_Loss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        # Using Euclidean Distance as per the paper
        dist_mat = torch.cdist(embeddings, embeddings, p=2)

        is_pos = labels.view(-1, 1).eq(labels.view(1, -1)).float()
        is_neg = labels.view(-1, 1).ne(labels.view(1, -1)).float()

        # Hardest Positive
        mask_pos = dist_mat * is_pos
        hardest_pos_dist = torch.max(mask_pos, dim=1)[0]

        # Hardest Negative
        # Add a large value to same-id distances to ignore them in min()
        mask_neg = dist_mat + (is_pos * 1e6)
        hardest_neg_dist = torch.min(mask_neg, dim=1)[0]

        loss = F.relu(hardest_pos_dist - hardest_neg_dist + self.margin)
        return loss.mean()
