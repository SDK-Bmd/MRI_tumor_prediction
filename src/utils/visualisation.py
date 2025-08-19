import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
from PIL import Image
import cv2
import os


def plot_training_history(history, save_path=None, title_prefix=''):
    """
    Plot training and validation metrics

    Args:
        history: Training history dictionary
        save_path: Path to save the figure
        title_prefix: Prefix for plot titles
    """
    plt.figure(figsize=(12, 10))

    # Plot accuracy
    plt.subplot(2, 2, 1)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title(f'{title_prefix}Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    # Plot loss
    plt.subplot(2, 2, 2)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title(f'{title_prefix}Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    # Plot learning rate if available
    if 'lr' in history:
        plt.subplot(2, 2, 3)
        plt.plot(history['lr'])
        plt.title(f'{title_prefix}Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')

    # Plot train/val accuracy gap (to visualize overfitting)
    if 'train_acc' in history and 'val_acc' in history:
        plt.subplot(2, 2, 4)
        acc_gap = np.array(history['train_acc']) - np.array(history['val_acc'])
        plt.plot(acc_gap, label='Train-Val Gap')
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        plt.title(f'{title_prefix}Train-Validation Accuracy Gap')
        plt.xlabel('Epoch')
        plt.ylabel('Gap (Train-Val)')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_confusion_matrix(cm, class_names, save_path=None, title='Confusion Matrix'):
    """
    Plot confusion matrix

    Args:
        cm: Confusion matrix array
        class_names: List of class names
        save_path: Path to save the figure
        title: Plot title
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_roc_curve(fpr, tpr, roc_auc, save_path=None, title='Receiver Operating Characteristic'):
    """
    Plot ROC curve

    Args:
        fpr: False positive rate array
        tpr: True positive rate array
        roc_auc: ROC AUC value
        save_path: Path to save the figure
        title: Plot title
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")

    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_sample_predictions(model, test_loader, device, class_names=None, num_samples=10, save_path=None):
    """
    Plot sample predictions

    Args:
        model: Trained PyTorch model
        test_loader: Test data loader
        device: Device to evaluate on
        class_names: List of class names
        num_samples: Number of samples to show
        save_path: Path to save the figure
    """
    if class_names is None:
        class_names = ["No Tumor", "Meningioma"]

    # Get batch of images
    dataiter = iter(test_loader)
    images, labels = next(dataiter)

    # Make predictions
    model.eval()
    with torch.no_grad():
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        predicted = (outputs > 0.5).float()

    # Limit to requested number of samples
    num_samples = min(num_samples, len(images))

    # Convert images for display
    images = images.cpu().numpy()
    images = np.transpose(images, (0, 2, 3, 1))

    # Denormalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    images = std * images + mean
    images = np.clip(images, 0, 1)

    # Create figure
    plt.figure(figsize=(15, 8))

    for i in range(num_samples):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i])

        # Get predicted and true classes
        pred_class = class_names[1] if predicted[i] > 0.5 else class_names[0]
        true_class = class_names[1] if labels[i] == 1 else class_names[0]
        confidence = outputs[i].item() if predicted[i] > 0.5 else 1 - outputs[i].item()

        # Color based on correct/incorrect
        color = 'green' if predicted[i] == labels[i] else 'red'
        title = f"Pred: {pred_class}\nTrue: {true_class}\nConf: {confidence:.2f}"

        plt.title(title, color=color)
        plt.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_augmentation_examples(image_path, augmentation_transform, num_examples=5, save_path=None):
    """
    Plot examples of data augmentation

    Args:
        image_path: Path to sample image
        augmentation_transform: Augmentation transforms
        num_examples: Number of examples to show
        save_path: Path to save the figure
    """
    # Load image
    image = Image.open(image_path).convert('RGB')

    # Apply augmentation multiple times
    augmented_images = [augmentation_transform(image) for _ in range(num_examples)]

    # Convert tensor images to numpy
    np_images = []
    for img in augmented_images:
        # Convert tensor to numpy and transpose
        np_img = img.numpy().transpose((1, 2, 0))

        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        np_img = std * np_img + mean
        np_img = np.clip(np_img, 0, 1)

        np_images.append(np_img)

    # Original image resized for comparison
    original_resized = image.resize((np_images[0].shape[1], np_images[0].shape[0]))
    original_np = np.array(original_resized) / 255.0

    # Plot images
    plt.figure(figsize=(12, 8))

    # Original image
    plt.subplot(2, 3, 1)
    plt.imshow(original_np)
    plt.title('Original')
    plt.axis('off')

    # Augmented images
    for i in range(num_examples):
        plt.subplot(2, 3, i + 2)
        plt.imshow(np_images[i])
        plt.title(f'Augmented {i + 1}')
        plt.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")

    plt.show()