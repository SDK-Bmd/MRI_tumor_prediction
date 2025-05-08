# Brain MRI Tumor Classification

A deep learning project for classifying brain MRI scans as normal or containing meningioma tumors, with a focus on addressing overfitting through enhanced data augmentation and early stopping tuning.

## Project Structure

- `src/`: Core implementation modules
  - `models/`: Neural network architectures
  - `data/`: Data loading and enhanced augmentation
  - `training/`: Training with improved early stopping
  - `utils/`: Visualization and model utilities
- `app/`: Streamlit web application
- `notebooks/`: Analysis notebooks including overfitting solutions
- `main.py`: Entry point script for training and evaluation

## Overfitting Solution

This project demonstrates how to address overfitting in medical imaging classification through:

1. **Enhanced Data Augmentation**: Implementing a robust augmentation pipeline with vertical flips, shear transformations, gamma adjustments, and noise to better simulate real-world variations in MRI scans.

2. **Early Stopping Tuning**: Finding the optimal patience parameter to stop training before the model overfits to the training data.

## Usage

### Training with Enhanced Augmentation

```bash
python main.py --train_dir datasets/Training --test_dir datasets/Testing --enhanced_aug --patience 15 --version V6_ENHANCED
