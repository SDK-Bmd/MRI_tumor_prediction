import torch
import torchvision.transforms as transforms
import numpy as np
import random
from PIL import ImageFilter, ImageOps


class GaussianNoise:
    """Add Gaussian noise to image"""

    def __init__(self, mean=0., std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean


class RandomGamma:
    """Apply random gamma correction"""

    def __init__(self, gamma_range=(0.8, 1.2)):
        self.gamma_range = gamma_range

    def __call__(self, img):
        gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
        return transforms.functional.adjust_gamma(img, gamma)


def get_enhanced_transforms(image_size=150, phase='train'):
    """
    Get enhanced data augmentation transforms

    Args:
        image_size: Size to resize images to
        phase: 'train', 'val', or 'test'

    Returns:
        torchvision transforms composition
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if phase == 'train':
        # Significantly enhanced training augmentations to prevent overfitting
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(degrees=30),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),  # Added vertical flip
            transforms.RandomAffine(
                degrees=0,
                translate=(0.2, 0.2),
                scale=(0.8, 1.2),
                shear=15  # Added shear transformation
            ),
            transforms.ColorJitter(
                brightness=0.3,  # Increased from 0.2
                contrast=0.3,  # Increased from 0.2
                saturation=0.2,  # New parameter
                hue=0.1  # New parameter
            ),
            RandomGamma(gamma_range=(0.8, 1.2)),  # Custom gamma adjustment
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3)  # Simulate lower quality scans
            ], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomApply([GaussianNoise(0, 0.05)], p=0.2)  # Add noise post-normalization
        ])
    else:
        # Validation and test transforms - only resize and normalize
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])