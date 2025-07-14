import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Configuration
TRAIN_IMAGE_DIR = "/media/jag/volD/JagguBhai/crack_dataset/train_rgb"
TRAIN_MASK_DIR = "/media/jag/volD/JagguBhai/crack_dataset/train_mask"
VAL_IMAGE_DIR = "/media/jag/volD/JagguBhai/crack_dataset/valid_rgb"
VAL_MASK_DIR = "/media/jag/volD/JagguBhai/crack_dataset/valid_mask"
TEST_IMAGE_DIR = "/media/jag/volD/JagguBhai/crack_dataset/test_rgb"

IMAGE_SIZE = (352, 352)
BATCH_SIZE = 2
NUM_WORKERS = 4

class CrackSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, augment=False):
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".jpg")])
        self.mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(".png")])
        self.augment = augment
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(IMAGE_SIZE),
        ])
        
        if augment:
            self.aug_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load and preprocess image
        image = cv2.imread(self.image_paths[idx])
        image = cv2.resize(image, IMAGE_SIZE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()
        
        # Load and preprocess mask
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, IMAGE_SIZE).copy()
        
        # Convert to tensors
        image = self.transform(image)
        mask = torch.from_numpy(mask / 255.0).float().unsqueeze(0)
        
        # Apply augmentations
        if self.augment:
            image = self.aug_transform(image)
        
        # Get filename without extension
        filename = os.path.basename(self.image_paths[idx]).replace('.jpg', '')
            
        return image, mask, filename

class CrackTestDataset(Dataset):
    def __init__(self, image_dir):
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".jpg")])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(IMAGE_SIZE),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.resize(image, IMAGE_SIZE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()
        image = self.transform(image)
        filename = os.path.basename(self.image_paths[idx]).replace('.jpg', '')
        return image, filename

def get_dataloaders():
    """Create and return train, validation, and test data loaders"""
    train_dataset = CrackSegmentationDataset(TRAIN_IMAGE_DIR, TRAIN_MASK_DIR, augment=True)
    val_dataset = CrackSegmentationDataset(VAL_IMAGE_DIR, VAL_MASK_DIR, augment=False)
    test_dataset = CrackTestDataset(TEST_IMAGE_DIR)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    return train_loader, val_loader, test_loader