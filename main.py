import argparse
import torch
import os
import numpy as np
import random
from src.models.cnn_model import MeningiomaModel
from src.models.transfer_learning import create_transfer_learning_model
from src.data.dataset import create_data_loaders
from src.data.augmentation import get_enhanced_transforms
from src.training.train import train_model
from src.training.evaluate import evaluate_model
from src.training.hyperparameter_tuning import tune_early_stopping
from src.utils.visualisation import plot_training_history, plot_confusion_matrix, plot_roc_curve, plot_sample_predictions, plot_augmentation_examples

from src.utils.model_utils import save_model_info, save_torchscript_model


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='MRI Tumor Detection Training Script')

    # Dataset parameters
    parser.add_argument('--train_dir', type=str, default='datasets/Training', help='Training data directory')
    parser.add_argument('--test_dir', type=str, default='datasets/Testing', help='Test data directory')
    parser.add_argument('--image_size', type=int, default=150, help='Input image size')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')

    # Model parameters
    parser.add_argument('--model_type', type=str, default='cnn', choices=['cnn', 'transfer'],
                        help='Model architecture (cnn or transfer learning)')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate for FC layers')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=25, help='Early stopping patience')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')

    # Augmentation parameters
    parser.add_argument('--enhanced_aug', action='store_true', help='Use enhanced augmentation')

    # Tuning parameters
    parser.add_argument('--tune_patience', action='store_true', help='Tune early stopping patience')

    # Output parameters
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    parser.add_argument('--version', type=str, default='V6', help='Model version')

    # Other parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create data loaders
    train_loader, val_loader, test_loader, class_names = create_data_loaders(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        use_enhanced_augmentation=args.enhanced_aug,
        seed=args.seed
    )

    # Print dataset information
    print(f"Class names: {class_names}")

    # Tune early stopping patience if requested
    if args.tune_patience:
        print("\n=== Tuning Early Stopping Patience ===")
        patience_values = [5, 10, 15, 20, 25, 30]

        results, best_patience = tune_early_stopping(
            train_loader=train_loader,
            val_loader=val_loader,
            model_type=args.model_type,
            patience_values=patience_values,
            image_size=args.image_size,
            epochs=args.epochs,
            device=device,
            save_dir=os.path.join(args.output_dir, 'tuning')
        )

        # Use best patience for training
        args.patience = best_patience
        print(f"Using best patience value: {args.patience}")

    # Create model
    if args.model_type == 'cnn':
        model = MeningiomaModel(image_size=args.image_size, fc_dropout=args.dropout).to(device)
        print("Using custom CNN model")
    else:
        model = create_transfer_learning_model(dropout_rate=args.dropout).to(device)
        print("Using ResNet18 with transfer learning")

    # Print model summary
    print("\nModel architecture:")
    print(model)

    # Count parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params / total_params:.2%})")

    # Define optimizer, criterion, and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    criterion = torch.nn.BCELoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=max(3, args.patience // 5), verbose=True
    )

    # Train model
    print("\n=== Starting Model Training ===")

    model_save_path = os.path.join(
        args.output_dir,
        f"meningioma_model_{args.version}{'_enhanced' if args.enhanced_aug else ''}.pth"
    )

    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.epochs,
        patience=args.patience,
        early_stopping_delta=0.001,
        model_save_path=model_save_path,
        version=args.version,
        verbose=True
    )

    # Plot training history
    history_plot_path = os.path.join(
        args.output_dir,
        f"training_history_{args.version}{'_enhanced' if args.enhanced_aug else ''}.png"
    )

    plot_training_history(
        history,
        save_path=history_plot_path,
        title_prefix=f"{args.model_type.upper()} "
    )

    # Evaluate model if test data available
    if test_loader is not None:
        print("\n=== Evaluating Model on Test Data ===")

        results = evaluate_model(
            model=model,
            test_loader=test_loader,
            device=device,
            class_names=class_names
        )

        # Plot confusion matrix
        cm_plot_path = os.path.join(
            args.output_dir,
            f"confusion_matrix_{args.version}{'_enhanced' if args.enhanced_aug else ''}.png"
        )

        plot_confusion_matrix(
            cm=results['confusion_matrix'],
            class_names=class_names,
            save_path=cm_plot_path,
            title=f"Confusion Matrix - {args.model_type.upper()}"
        )

        # Plot ROC curve
        roc_plot_path = os.path.join(
            args.output_dir,
            f"roc_curve_{args.version}{'_enhanced' if args.enhanced_aug else ''}.png"
        )

        plot_roc_curve(
            fpr=results['fpr'],
            tpr=results['tpr'],
            roc_auc=results['roc_auc'],
            save_path=roc_plot_path,
            title=f"ROC Curve - {args.model_type.upper()}"
        )

        # Plot sample predictions
        samples_plot_path = os.path.join(
            args.output_dir,
            f"sample_predictions_{args.version}{'_enhanced' if args.enhanced_aug else ''}.png"
        )

        plot_sample_predictions(
            model=model,
            test_loader=test_loader,
            device=device,
            class_names=class_names,
            num_samples=10,
            save_path=samples_plot_path
        )

    # Save model info
    model_info_path = os.path.join(
        args.output_dir,
        f"model_info_{args.version}{'_enhanced' if args.enhanced_aug else ''}.json"
    )

    class_mapping = {str(i): name for i, name in enumerate(class_names)}

    save_model_info(
        model=model,
        model_path=model_save_path,
        info_path=model_info_path,
        image_size=args.image_size,
        version=args.version,
        class_mapping=class_mapping
    )

    # Save TorchScript model
    torchscript_path = os.path.join(
        args.output_dir,
        f"meningioma_model_{args.version}{'_enhanced' if args.enhanced_aug else ''}_torchscript.pt"
    )

    save_torchscript_model(
        model=model,
        output_path=torchscript_path,
        image_size=args.image_size,
        device=device
    )

    print(f"\nComplete! Model saved to {model_save_path} and prepared for deployment at {torchscript_path}")


if __name__ == "__main__":
    main()