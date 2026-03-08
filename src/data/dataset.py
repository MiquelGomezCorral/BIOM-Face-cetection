import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from maikol_utils.print_utils import print_separator
from maikol_utils.file_utils import list_dir_files

from src.config import Configuration
from .filter import local_normalize_image

class FACES_DATASET(Dataset):
    def __init__(self, partition = "train", transform = None, CONFIG: Configuration = None):
        self.partition = partition
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.ToTensor(),
        ])
        self.config = CONFIG

        if self.partition == "train":
            self.data_paths, self.n = list_dir_files(self.config.train_path)
        elif self.partition == "val":
            self.data_paths, self.n = list_dir_files(self.config.val_path)
        else:
            self.data_paths, self.n = list_dir_files(self.config.test_path)

        print(f" - Total data {self.partition}: {self.n} images")

    
    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if self.config.gray_scale:
            img = cv2.imread(self.data_paths[idx], cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(self.data_paths[idx])
        
        # Filter for the images
        img = local_normalize_image(self.config, img)
        
        # Clip and convert to uint8
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        # Resize
        img = cv2.resize(img, (self.config.crop_size, self.config.crop_size), interpolation=cv2.INTER_AREA)
        
        # Convert to PIL for transforms
        img = Image.fromarray(img)

        # data augmentation
        img_tensor = self.transform(img)

        # Label
        label = torch.tensor('Human' in self.data_paths[idx], dtype=torch.long)
        return {"img": img_tensor, "label": label}
    


def load_faces(CONFIG: Configuration):
    print_separator(f"Loading FACES Dataset...")
    
    train_da = transforms.Compose([
        transforms.RandomHorizontalFlip(p=CONFIG.aug_prob),
        transforms.RandomApply([
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2)
        ], p=CONFIG.aug_prob),
        transforms.ToTensor(),
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = FACES_DATASET(partition="train", transform=train_da, CONFIG=CONFIG)
    val_dataset = FACES_DATASET(partition="val", transform=train_da, CONFIG=CONFIG)
    test_dataset = FACES_DATASET(partition="test", transform=test_transform, CONFIG=CONFIG)

    # DataLoader Class
    train_dataloader = DataLoader(train_dataset, CONFIG.batch_size, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, CONFIG.batch_size, shuffle=False, num_workers=0)
    test_dataloader = DataLoader(test_dataset, CONFIG.batch_size, shuffle=False, num_workers=0)

    return train_dataloader, val_dataloader, test_dataloader
