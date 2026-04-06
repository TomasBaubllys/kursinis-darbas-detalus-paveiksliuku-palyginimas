# Author: Tomas Baublys

import os
import random
from collections import defaultdict

import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

DEFAULT_DATA_PATH = "../data/Market-1501-v15.09.15/bounding_box_train"


class Market_Train_Dataset(Dataset):
    def __init__(self, root_dir=DEFAULT_DATA_PATH, transform=None, id_list=None):
        dataset_dir_train = root_dir
        full_dataset_dir = os.path.join(os.getcwd(), dataset_dir_train)

        if not os.path.exists(full_dataset_dir):
            raise FileNotFoundError(f"Directory {full_dataset_dir} not found.")

        all_files = os.listdir(full_dataset_dir)
        self.id_to_images = defaultdict(list)
        self.transform = transform

        for f in all_files:
            if f.endswith(".jpg"):
                parts = f.split("_")
                person_id = parts[0]
                cam_id = parts[1]
                if person_id not in ["-1", "0000"]:
                    self.id_to_images[person_id].append(
                        (os.path.join(full_dataset_dir, f), cam_id)
                    )

        if id_list is not None:
            self.person_ids = sorted(list(id_list))
        else:
            self.person_ids = sorted(list(self.id_to_images.keys()))

        self.id_to_label = {pid: i for i, pid in enumerate(self.person_ids)}

        self.num_ids = len(self.person_ids)
        self.transform = transform

    def __len__(self):
        return len(self.person_ids)

    def __getitem__(self, idx):
        target_id = self.person_ids[idx]
        img_path, cam_id = random.choice(self.id_to_images[target_id])
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, self.id_to_label[target_id], cam_id


class Market_Eval_Dataset(Dataset):
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
