# Author: Tomas Baublys

import torch
import torch.nn as nn
import torch.nn.functional as F


class Batch_Hard_Triplet_Loss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        # calc the pairwise distance
        dist_mat = torch.cdist(embeddings, embeddings, p=2)

        # create masks for same/diff ids
        is_pos = labels.view(-1, 1).eq(labels.view(1, -1)).float()
        is_neg = labels.view(-1, 1).ne(labels.view(1, -1)).float()

        mask_pos = dist_mat * is_pos
        hardest_pos_dist = torch.max(mask_pos, dim=1)[0]

        mask_neg = dist_mat + (is_pos * 1e6)
        hardest_neg_dist = torch.min(mask_neg, dim=1)[0]

        # max(0, d(a, p) - d(a, n) + margin)
        loss = F.relu(hardest_pos_dist - hardest_neg_dist + self.margin)
        return loss.mean()
