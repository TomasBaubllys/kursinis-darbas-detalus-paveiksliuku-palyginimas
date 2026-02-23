import os
from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset
import random
import matplotlib.pyplot as plt
from torchvision import transforms


class MarketSiameseDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        dataset_dir_train = '../data/Market-1501-v15.09.15/bounding_box_train'
        full_dataset_dir_train = os.path.join(os.getcwd(), dataset_dir_train)

        all_files = os.listdir(full_dataset_dir_train)
        self.id_to_images = defaultdict(list)
        self.transform = transform

        for f in all_files:
            if f.endswith('.jpg'):
                person_id = f.split('_')[0]
                if person_id not in ['-1', '0000']:
                    self.id_to_images[person_id].append(os.path.join(full_dataset_dir_train, f))

        self.person_ids = list(self.id_to_images.keys())

    def __len__(self):
        return len(self.person_ids)

    def __getitem__(self, idx):
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

        img1 = Image.open(img_path1).convert('RGB')
        img2 = Image.open(img_path2).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label

# AI GENERATED CODE BELOW
def test_siamese_dataset(data_path):
    # 1. Define a basic transform (No normalization yet so the colors look normal in the plot)
    test_transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor()
    ])

    # 2. Initialize your class
    dataset = MarketSiameseDataset(root_dir=data_path, transform=test_transform)

    # 3. Create a figure to visualize 3 pairs
    fig, axes = plt.subplots(3, 2, figsize=(10, 15))

    for i in range(3):
        # This triggers your __getitem__ logic
        img1, img2, label = dataset[i]

        # Convert tensors back to images for plotting (C, H, W -> H, W, C)
        img1_np = img1.permute(1, 2, 0).numpy()
        img2_np = img2.permute(1, 2, 0).numpy()

        # Plotting
        axes[i, 0].imshow(img1_np)
        axes[i, 0].set_title(f"Pair {i+1}: Image A")
        axes[i, 1].imshow(img2_np)
        axes[i, 1].set_title(f"Pair {i+1}: Image B (Label: {label})")

        print(f"Sample {i+1}: Label {label} ({'Same' if label == 1.0 else 'Different'} Person)")

    plt.tight_layout()
    plt.savefig("lol.jpg")

if __name__ == "__main__":
    # Update this path to your actual folder
    train_path = '../data/Market-1501-v15.09.15/bounding_box_train'

    # Run your index check
    # (Assuming your index_training_data() logic is now inside the class or available)

    # Run the visual test
    test_siamese_dataset(train_path)
