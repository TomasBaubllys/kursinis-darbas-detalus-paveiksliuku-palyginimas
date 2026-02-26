# Author: Tomas Baublys

import os
import random
from collections import defaultdict

import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

DEFAULT_DATA_PATH = "../data/Market-1501-v15.09.15/bounding_box_train"


class MarketSiameseDataset(Dataset):
    def __init__(self, root_dir=DEFAULT_DATA_PATH, transform=None):
        dataset_dir_train = root_dir
        full_dataset_dir_train = os.path.join(os.getcwd(), dataset_dir_train)

        all_files = os.listdir(full_dataset_dir_train)
        self.id_to_images = defaultdict(list)
        self.transform = transform

        for f in all_files:
            if f.endswith(".jpg"):
                person_id = f.split("_")[0]
                if person_id not in ["-1", "0000"]:
                    self.id_to_images[person_id].append(
                        os.path.join(full_dataset_dir_train, f)
                    )

        self.person_ids = sorted(list(self.id_to_images.keys()))

        # NEW: Create a mapping from ID string to 0-indexed integer
        # e.g., "0002" -> 0, "0007" -> 1, "1501" -> 750
        self.id_to_label = {pid: i for i, pid in enumerate(self.person_ids)}

        # NEW: Attribute for your training script to find
        self.num_ids = len(self.person_ids)
        self.transform = transform

    def __len__(self):
        return len(self.person_ids)

    def __getitem__(self, idx):
        target_id = self.person_ids[idx]
        img_path = random.choice(self.id_to_images[target_id])
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        # UPDATED: Return the mapped integer label, not the raw ID
        return img, self.id_to_label[target_id]

    # not used in the actual training, since we switched to triplet loss
    def __getitem__contrastive_loss(self, idx):
        target_id = self.person_ids[idx]

        # decide if we want the same person or not
        should_get_same_person = random.randint(0, 1)

        img_path1 = random.choice(self.id_to_images[target_id])

        if should_get_same_person:
            img_path2 = random.choice(self.id_to_images[target_id])
            # 1.0 means its the same image
            label = 1.0
        else:
            external_id = random.choice(self.person_ids)
            while external_id == target_id:
                external_id = random.choice(self.person_ids)
            img_path2 = random.choice(self.id_to_images[external_id])
            # 0.0 means its a different image
            label = 0.0

        img1 = Image.open(img_path1).convert("RGB")
        img2 = Image.open(img_path2).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label
