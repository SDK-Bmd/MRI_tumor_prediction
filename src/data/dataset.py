import os
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets
import numpy as np
import random
from PIL import Image
from .augmentation import get_enhanced_transforms


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MRIDataset(Dataset):
    """Custom dataset for MRI images with enhanced loading capabilities"""

    def __init__(self, root_dir, transform=None, class_to_idx=None):
        """
        Args:
            root_dir: Directory with class subdirectories
            transform: Optional transform to apply
            class_to_idx: Optional class mapping
        """
        self.root_dir = root_dir
        self.transform = transform

        # Find all image files
        self.samples = []
        self.classes = []

        # Use provided class mapping or auto-detect
        if class_to_idx is None:
            class_to_idx = {}
            idx = 0
            for item in sorted(os.listdir(root_dir)):
                if os.path.isdir(os.path.join(root_dir, item)):
                    class_to_idx[item] = idx
                    idx += 1

        self.class_to_idx = class_to_idx

        # Walk through directories
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                continue

            self.classes.append(class_name)

            # Get all valid image files
            for file_name in os.listdir(class_dir):
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                    self.samples.append({
                        "path": os.path.join(class_dir, file_name),
                        "class_idx": class_idx,
                        "class_name": class_name
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample["path"]).convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, sample["class_idx"]


def create_data_loaders(
        train_dir,
        test_dir=None,
        image_size=150,
        batch_size=16,
        val_split=0.2,
        num_workers=4,
        use_enhanced_augmentation=True,
        seed=42
):
    """
    Create training, validation, and test data loaders

    Args:
        train_dir: Training data directory
        test_dir: Test data directory (optional)
        image_size: Size to resize images to
        batch_size: Batch size for training
        val_split: Validation split ratio
        num_workers: Number of worker threads for data loading
        use_enhanced_augmentation: Whether to use enhanced augmentation
        seed: Random seed

    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    set_seed(seed)

    # Get transforms
    if use_enhanced_augmentation:
        print("Using enhanced data augmentation pipeline...")
        train_transform = get_enhanced_transforms(image_size, 'train')
    else:
        # Basic augmentation (original version)
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(0, translate=(0.2, 0.2), scale=(0.8, 1.2)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # Validation and test transform
    val_test_transform = get_enhanced_transforms(image_size, 'val')

    # Create training dataset
    train_dataset = datasets.ImageFolder(
        train_dir,
        transform=train_transform
    )

    # Print class mapping
    print(f"Class mapping: {train_dataset.class_to_idx}")

    # Split into train and validation
    val_size = int(val_split * len(train_dataset))
    train_size = len(train_dataset) - val_size

    train_subset, val_subset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    # Apply validation transform to validation dataset
    val_dataset = datasets.ImageFolder(
        train_dir,
        transform=val_test_transform
    )

    # Use same indices from val_subset
    val_subset = torch.utils.data.Subset(val_dataset, val_subset.indices)

    # Create data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Create test loader if test directory provided
    test_loader = None
    if test_dir and os.path.exists(test_dir):
        test_dataset = datasets.ImageFolder(
            test_dir,
            transform=val_test_transform
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    # Get class names
    class_names = [c.split('-')[-1] for c in sorted(train_dataset.classes)]

    # Log dataset sizes
    print(f"Training set size: {len(train_subset)}")
    print(f"Validation set size: {len(val_subset)}")
    if test_loader:
        print(f"Test set size: {len(test_loader.dataset)}")

    return train_loader, val_loader, test_loader, class_names