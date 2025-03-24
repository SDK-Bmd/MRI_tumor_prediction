import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os
import random

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Configuration settings
IMAGE_SIZE = 150  # Resize images to this size
BATCH_SIZE = 7
EPOCHS = 50
PATIENCE = 25
LEARNING_RATE = 0.001
VERSION = "V5_TORCH"
MODEL_PATH = "meningioma_model_" + VERSION + ".pth"

# Paths to your data
TRAIN_DIR = "datasets/Training"
TEST_DIR = "datasets/Testing"

# Check for GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def verify_dataset_structure():
    """Verify dataset structure and count images"""
    dataset_stats = {}
    root_folder = ["Training", "Testing"]
    sub_folder = ["0-notumor", "1-meningioma"]

    for dataset_type in root_folder:
        dataset_stats[dataset_type] = {}
        path = os.path.join("datasets", dataset_type)

        for class_name in sub_folder:
            class_path = os.path.join(path, class_name)
            if not os.path.exists(class_path):
                print(f"ERROR: Path {class_path} does not exist!")
                continue

            # Count images in folder
            image_count = len([f for f in os.listdir(class_path)
                               if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))])
            dataset_stats[dataset_type][class_name] = image_count

    # Print statistics
    print("\n==== Dataset Statistics ====")
    for dataset_type, stats in dataset_stats.items():
        print(f"{dataset_type} set:")
        for class_name, count in stats.items():
            print(f"  - {class_name}: {count} images")
        total = sum(stats.values())
        print(f"  Total: {total} images")

    return dataset_stats


def create_data_loaders():
    """Create and return training, validation, and test data loaders"""
    print("\nCreating data loaders...")

    # Data augmentation for training set
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(0, translate=(0.2, 0.2), scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Additional augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Only resize and normalize for validation and test sets
    val_test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = datasets.ImageFolder(
        os.path.join(TRAIN_DIR),
        transform=train_transform
    )

    # Print default class mapping
    print(f"Default class mapping from ImageFolder: {train_dataset.class_to_idx}")

    # Split train dataset into train and validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Fixed seed for reproducible splits
    )

    # Apply validation transform to validation dataset
    val_dataset = datasets.ImageFolder(
        os.path.join(TRAIN_DIR),
        transform=val_test_transform
    )

    # Use the same indices from val_subset
    indices_val = val_subset.indices
    val_subset = torch.utils.data.Subset(val_dataset, indices_val)

    # Create test dataset with the same class mapping
    test_dataset = datasets.ImageFolder(
        os.path.join(TEST_DIR),
        transform=val_test_transform
    )

    # Print test dataset's default mapping
    print(f"Test dataset default mapping: {test_dataset.class_to_idx}")

    # Create data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Verify class distribution in all datasets
    print("\nVerifying class distribution in datasets:")
    for name, loader in [("Training", train_loader), ("Validation", val_loader), ("Testing", test_loader)]:
        class_counts = {0: 0, 1: 0}
        for _, labels in loader:
            for label in labels:
                class_counts[label.item()] += 1
        print(f"{name} set - Class 0 (notumor): {class_counts[0]}, Class 1 (meningioma): {class_counts[1]}")

    # Class names need to be sorted by index for correct reporting
    class_names = ['notumor', 'meningioma']
    print(f"Class names for reporting: {class_names}")

    return train_loader, val_loader, test_loader, class_names


class MeningiomaModel(nn.Module):
    """CNN model architecture for meningioma detection"""

    def __init__(self):
        super(MeningiomaModel, self).__init__()

        # First convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )

        # Second convolutional block
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )

        # Third convolutional block
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )

        # Calculate the size after convolving and pooling
        conv_output_size = IMAGE_SIZE // 8

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * conv_output_size * conv_output_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc(x)
        return x


def create_transfer_learning_model():
    """Create a model using transfer learning with ResNet18"""
    print("Creating transfer learning model with ResNet18...")

    # Load pre-trained ResNet18
    model = models.resnet18(pretrained=True)

    # Freeze early layers
    for param in list(model.parameters())[:-8]:  # Only train the last few layers
        param.requires_grad = False

    # Replace final layer for binary classification
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 1),
        nn.Sigmoid()
    )

    return model


def train_model(model, train_loader, val_loader, criterion, optimizer):
    """Train the model and return training history"""
    print("\nStarting model training...")

    # Track best validation accuracy and loss
    best_val_loss = float('inf')
    no_improve_epochs = 0

    # Set up learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # History for plotting
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.float().to(device)
            labels = labels.view(-1, 1)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            train_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_train_acc = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.float().to(device)
                labels = labels.view(-1, 1)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Statistics
                val_loss += loss.item() * inputs.size(0)
                predicted = (outputs > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = val_correct / val_total

        # Update learning rate scheduler
        scheduler.step(epoch_val_loss)

        # Save history for plotting
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        # Print statistics
        print(f'Epoch {epoch + 1}/{EPOCHS} | '
              f'Train Loss: {epoch_train_loss:.4f} | '
              f'Train Acc: {epoch_train_acc:.4f} | '
              f'Val Loss: {epoch_val_loss:.4f} | '
              f'Val Acc: {epoch_val_acc:.4f}')

        # Save best model and check for early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            no_improve_epochs = 0
            print(f"✓ Model saved with Val Loss: {best_val_loss:.4f}")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= PATIENCE:
                print(f"Early stopping triggered after epoch {epoch + 1}")
                break

    # Load best model
    model.load_state_dict(torch.load(MODEL_PATH))
    return model, history


def evaluate_model(model, test_loader, class_names):
    """Evaluate the model and display performance metrics"""
    print("\nEvaluating model on test data...")
    model.eval()

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            probs = outputs.cpu().numpy()
            predicted = (outputs > 0.5).float().cpu().numpy()

            # Collect predictions and labels
            all_probs.extend(probs)
            all_preds.extend(predicted)
            all_labels.extend(labels.cpu().numpy())

    # Convert to numpy arrays
    y_prob = np.array(all_probs).flatten()
    y_pred = np.array(all_preds).flatten()
    y_true = np.array(all_labels)

    # Calculate accuracy
    test_accuracy = np.mean(y_pred.round() == y_true)
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Classification report
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred.round(), target_names=class_names)
    print(report)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred.round())
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix_' + VERSION + '.png')
    plt.show()

    # ROC Curve and AUC
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve_' + VERSION + '.png')
    plt.show()

    return y_true, y_pred, y_prob


def plot_training_history(history):
    """Plot training and validation accuracy/loss"""
    print("\nPlotting training history...")

    # Plot accuracy
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history_' + VERSION + '.png')
    plt.show()


def save_torchscript_model(model):
    """Save the model in TorchScript format for deployment"""
    print("\nSaving model for deployment...")

    # Ensure model is in evaluation mode
    model.eval()

    # Create example input
    example_input = torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)

    # Export to TorchScript
    scripted_model = torch.jit.trace(model, example_input)
    torchscript_path = "meningioma_model_" + VERSION + "_torchscript.pt"
    scripted_model.save(torchscript_path)
    print(f"Model saved in TorchScript format at: {torchscript_path}")

    # Also save model documentation with training info
    doc = {
        "model_version": VERSION,
        "input_size": [3, IMAGE_SIZE, IMAGE_SIZE],
        "EPOCH": EPOCHS,
        "LEARNING_RATE": LEARNING_RATE,
        "class_mapping": {"0": "notumor", "1": "meningioma"},
        "prediction_threshold": 0.5,
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        }
    }

    with open(f"model_info_{VERSION}.txt", "w") as f:
        for key, value in doc.items():
            f.write(f"{key}: {value}\n")

    return torchscript_path


def sample_predictions(model, test_loader, num_samples=10):
    """Show sample predictions for visualization"""
    print("\nGenerating sample predictions...")

    model.eval()
    samples = []

    # Get a batch of data
    dataiter = iter(test_loader)
    images, labels = next(dataiter)

    # Make predictions
    with torch.no_grad():
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        predicted = (outputs > 0.5).float()

    # Get the first 'num_samples' examples
    num_samples = min(num_samples, len(images))

    # Convert images back to CPU and denormalize for display
    images = images.cpu().numpy()

    # Transpose from (B, C, H, W) to (B, H, W, C) for plotting
    images = np.transpose(images, (0, 2, 3, 1))

    # Denormalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    images = std * images + mean
    images = np.clip(images, 0, 1)

    # Plot images with predictions
    plt.figure(figsize=(12, 8))

    for i in range(num_samples):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i])

        pred_class = "Meningioma" if predicted[i] > 0.5 else "No Tumor"
        true_class = "Meningioma" if labels[i] == 1 else "No Tumor"
        confidence = outputs[i].item() if predicted[i] > 0.5 else 1 - outputs[i].item()

        # Color based on correct/incorrect prediction
        color = 'green' if predicted[i] == labels[i] else 'red'
        title = f"Pred: {pred_class}\nTrue: {true_class}\nConf: {confidence:.2f}"

        plt.title(title, color=color)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('sample_predictions_' + VERSION + '.png')
    plt.show()


def main():

    print("=" * 50)
    print("MRI TUMOR DETECTION MODEL TRAINING")
    print("=" * 50)

    # First, verify the dataset structure
    dataset_stats = verify_dataset_structure()

    # Create data loaders
    train_loader, val_loader, test_loader, class_names = create_data_loaders()

    # Choose model architecture
    use_transfer_learning = input("\nUse transfer learning with ResNet18? (y/n): ").lower() == 'y'

    if use_transfer_learning:
        model = create_transfer_learning_model().to(device)
        print("Using ResNet18 with transfer learning")
    else:
        print("Building custom CNN model...")
        model = MeningiomaModel().to(device)

    print("\nModel architecture:")
    print(model)

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Print summary of trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params / total_params:.1%})")

    # Train the model
    print("\nStarting training...")
    model, history = train_model(model, train_loader, val_loader, criterion, optimizer)

    # Evaluate the model
    y_true, y_pred, y_prob = evaluate_model(model, test_loader, class_names)

    # Plot training history
    plot_training_history(history)

    # Show sample predictions
    sample_predictions(model, test_loader)

    # Save model for deployment
    torchscript_path = save_torchscript_model(model)

    print(f"\nComplete! Model saved to {MODEL_PATH} and prepared for deployment at {torchscript_path}")


if __name__ == "__main__":
    main()