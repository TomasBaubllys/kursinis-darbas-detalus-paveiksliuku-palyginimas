import random

from torch.utils.data import Sampler


class PKSampler(Sampler):
    def __init__(self, data_source, p=16, k=4):
        super().__init__
        self.data_source = data_source
        self.p = p
        self.k = k
        self.num_people = len(data_source)

    def __iter__(self):
        indices = []
        person_indices = list(range(self.num_people))
        random.shuffle(person_indices)

        for i in range(self.num_people // self.p):
            selected_people = person_indices[i * self.p : (i + 1) * self.p]

            for person_idx in selected_people:
                indices.extend([person_idx] * self.k)

        return iter(indices)

    def __len__(self):
        return (self.num_people // self.p) * self.p * self.k
