import random

from torch.utils.data import Sampler


class PKSampler(Sampler):
    def __init__(self, data_source, p=16, k=4):
        super().__init__()
        self.p = p
        self.k = k

        # Build pid_index -> [image_indices] map using the dataset's structure
        self.pid_to_indices = {}
        for pid_idx, pid in enumerate(data_source.person_ids):
            num_images = len(data_source.id_to_images[pid])
            self.pid_to_indices[pid_idx] = list(range(num_images))

        self.num_people = len(data_source.person_ids)
        self.data_source = data_source

    def __iter__(self):
        indices = []
        person_indices = list(range(self.num_people))
        random.shuffle(person_indices)

        for i in range(self.num_people // self.p):
            selected_people = person_indices[i * self.p : (i + 1) * self.p]

            for person_idx in selected_people:
                # Each call to __getitem__ with the same person_idx will
                # trigger random.choice inside the dataset, giving K different images
                indices.extend([person_idx] * self.k)

        return iter(indices)

    def __len__(self):
        return (self.num_people // self.p) * self.p * self.k
