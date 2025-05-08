import torch
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from tqdm import tqdm


def evaluate_model(model, test_loader, device, class_names=None):
    """
    Evaluate a trained model on test data

    Args:
        model: Trained PyTorch model
        test_loader: Test data loader
        device: Device to evaluate on
        class_names: List of class names

    Returns:
        Dictionary of evaluation metrics and predictions
    """
    model.eval()

    all_preds = []
    all_probs = []
    all_labels = []

    # Progress bar
    test_iterator = tqdm(test_loader, desc="Evaluating")

    with torch.no_grad():
        for inputs, labels in test_iterator:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)

            # Get probabilities and predictions
            probs = outputs.cpu().numpy()
            preds = (outputs > 0.5).float().cpu().numpy()

            # Store results
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    # Convert to numpy arrays
    all_probs = np.array(all_probs).flatten()
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()

    # Calculate metrics
    accuracy = np.mean(all_preds.round() == all_labels)
    cm = confusion_matrix(all_labels, all_preds.round())

    # Default class names if not provided
    if class_names is None:
        class_names = ["No Tumor", "Meningioma"]

    # Classification report
    report = classification_report(
        all_labels,
        all_preds.round(),
        target_names=class_names,
        output_dict=True
    )

    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    # Create results dictionary
    results = {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'roc_auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr,
        'predictions': all_preds,
        'probabilities': all_probs,
        'true_labels': all_labels,
        'class_names': class_names
    }

    # Print summary
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds.round(), target_names=class_names))

    return results