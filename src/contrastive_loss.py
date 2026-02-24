import torch
import torch.nn as nn

class Contrastive_Loss(nn.Module):
    def __init__(self, margin=2.0):
        super().__init__()
		self.margin = margin

	def forward(self, output1, output2, label):
		euclidean_dist = nn.pairwise_distance(output1, output2)

		pos = label * torch.pow(euclidean_dist, 2)

		neg = (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_dist, min = 0.0), 2)

		return torch.mean(pos + neg)




