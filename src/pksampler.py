import random

from torch.utils.data import Sampler


class PKSampler(Sampler):
    def __init__(self, data_source, p=16, k=4):
        super().__init__()
        self.data_source = data_source
        self.p = p
        self.k = k
        self.person_ids = data_source.person_ids
        self.id_to_indices = {pid: i for i, pid in enumerate(self.person_ids)}

    def __iter__(self):
        indices = []
        random.shuffle(self.person_ids)

        for i in range(len(self.person_ids) // self.p):
            selected_ids = self.person_ids[i * self.p : (i + 1) * self.p]
            for pid in selected_ids:
                # pick k images for the current pid
                idx = self.id_to_indices[pid]
                indices.extend([self.id_to_indices[idx]] * self.k)

        return iter(indices)

    def __len__(self):
        return len(self.person_ids)
