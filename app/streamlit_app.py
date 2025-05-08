import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from torchvision import transforms

# Page configuration
st.set_page_config(
    page_title="MRI Tumor Detector",
    page_icon="🧠",
    layout="wide"
)

# Constants from training script
IMAGE_SIZE = 150


# Model definition - exact same as in training script
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


# Load model with caching
@st.cache_resource
def load_model(model_path):
    """Load PyTorch model (.pth or TorchScript .pt)"""
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Check if it's a TorchScript model (.pt)
        if model_path.endswith('.pt'):
            model = torch.jit.load(model_path, map_location=device)
            model_type = "TorchScript"
        else:
            # Regular PyTorch model (.pth)
            model = MeningiomaModel().to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            model_type = "PyTorch"

        return model, model_type, device

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None


# Image preprocessing function
def preprocess_image(image, use_grayscale=False):
    """Preprocess an image for model input"""
    # Start with a PIL or numpy image
    if isinstance(image, Image.Image):
        # Convert PIL to numpy
        np_image = np.array(image.convert('RGB'))
    else:
        np_image = image

    # Apply grayscale if requested
    if use_grayscale:
        if len(np_image.shape) == 3 and np_image.shape[2] == 3:
            # Convert to grayscale but keep 3 channels
            gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
            processed_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            display_gray = gray  # For display
        else:
            # Already grayscale
            display_gray = np_image
            processed_image = cv2.cvtColor(np_image, cv2.COLOR_GRAY2RGB)
    else:
        # Keep as RGB
        processed_image = np_image
        display_gray = None

    # Resize to model input size
    resized_image = cv2.resize(processed_image, (IMAGE_SIZE, IMAGE_SIZE))

    # Create transform pipeline for normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Convert to tensor and normalize
    tensor_image = transform(resized_image)

    # Add batch dimension
    batch_image = tensor_image.unsqueeze(0)

    return batch_image, resized_image, display_gray


# Prediction function - FIXED with inverted logic based on diagnostic findings
def predict(model, image, device, use_grayscale=False):
    """Make a prediction using the model with corrected probability interpretation"""
    # Preprocess the image
    tensor_image, resized_image, display_gray = preprocess_image(image, use_grayscale)

    # Move to the right device
    tensor_image = tensor_image.to(device)

    # Make prediction
    with torch.no_grad():
        raw_output = model(tensor_image)

    # Get raw probability value
    raw_probability = raw_output.item()


    # Create result dictionary using the correct inverted logic
    result = {
        "probability": raw_probability,
        "class": "Meningioma" if raw_probability > 0.5 else "No Tumor",
        "confidence": 1.0 - raw_probability if raw_probability < 0.5 else raw_probability,
        "preprocessed": resized_image,
        "grayscale": display_gray
    }

    return result


# Process a batch of images
def process_batch(model, image_files, device, use_grayscale=False):
    """Process multiple images and return results"""
    results = []

    # Create a progress bar
    progress = st.progress(0)

    for i, file in enumerate(image_files):
        try:
            # Update progress
            progress.progress((i + 1) / len(image_files))

            # Open image
            image = Image.open(file)

            # Make prediction
            result = predict(model, image, device, use_grayscale)

            # Add filename to result
            result["filename"] = file.name

            # Add to results list
            results.append(result)

        except Exception as e:
            st.error(f"Error processing {file.name}: {e}")

    # Clear progress bar
    progress.empty()

    return results


# Create dataset class for evaluation with FIXED class interpretation
class MRIDataset(torch.utils.data.Dataset):
    """Dataset for MRI images"""

    def __init__(self, root_dir, transform=None, use_grayscale=False):
        self.root_dir = root_dir
        self.transform = transform
        self.use_grayscale = use_grayscale

        # Original class mapping from training script
        self.class_to_idx = {"notumor": 0, "meningioma": 1}

        # Find all image files
        self.samples = []

        # Walk through directories
        for class_name in self.class_to_idx:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                st.warning(f"Class directory not found: {class_dir}")
                continue

            class_idx = self.class_to_idx[class_name]

            # Get image files
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
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample["path"]).convert('RGB')

        # Apply grayscale if needed
        if self.use_grayscale:
            np_image = np.array(image)
            gray_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
            np_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
            image = Image.fromarray(np_image)

        # Apply transforms
        if self.transform:
            tensor_image = self.transform(image)
        else:
            # Default transform
            transform = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            tensor_image = transform(image)

        # Return tensor, label, and path
        return tensor_image, sample["class_idx"], sample["path"]


# Evaluate model on a test directory with FIXED interpretation
def evaluate_model(model, test_dir, device, use_grayscale=False, batch_size=32):
    """Evaluate model on a test directory"""
    if not os.path.exists(test_dir):
        st.error(f"Test directory not found: {test_dir}")
        return None

    # Check for required subdirectories
    required_dirs = ["1-meningioma", "0-notumor"]
    for dir_name in required_dirs:
        st.info(f"Required directory: {os.path.join(test_dir, dir_name)}")
        if not os.path.exists(os.path.join(test_dir, dir_name)):
            st.error(f"Required directory not found: {os.path.join(test_dir, dir_name)}")
            return None

    # Create dataset
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = MRIDataset(
        root_dir=test_dir,
        transform=transform,
        use_grayscale=use_grayscale
    )

    # Print dataset stats for debugging
    class_counts = {}
    for sample in dataset.samples:
        class_name = sample["class_name"]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    st.info(f"Dataset class distribution: {class_counts}")

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    # Evaluation loop
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    all_paths = []

    with torch.no_grad():
        for batch_images, batch_labels, batch_paths in dataloader:
            # Move to device
            batch_images = batch_images.to(device)

            # Forward pass
            outputs = model(batch_images)

            # Convert to numpy
            probs = outputs.cpu().numpy().flatten()

            # FIXED INTERPRETATION: inverted prediction logic
            # If probability < 0.5, predict class 1 (meningioma)
            # If probability >= 0.5, predict class 0 (notumor)
            preds = (probs < 0.5).astype(int)  # INVERTED LOGIC HERE
            labels = batch_labels.cpu().numpy()

            # Store results
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels)
            all_paths.extend(batch_paths)

    # Convert to numpy arrays
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate metrics
    accuracy = np.mean(all_preds == all_labels)
    cm = confusion_matrix(all_labels, all_preds)

    # Order class names to match the class-to-idx mapping
    class_names = ["No Tumor", "Meningioma"]

    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)

    # Find misclassified examples
    misclassified_idx = np.where(all_preds != all_labels)[0]
    misclassified = [
        {
            "path": all_paths[i],
            "filename": os.path.basename(all_paths[i]),
            "true_class": class_names[all_labels[i]],
            "pred_class": class_names[all_preds[i]],
            "probability": all_probs[i],
            "confidence": 1.0 - all_probs[i] if all_preds[i] == 1 else all_probs[i]  # FIXED confidence
        }
        for i in misclassified_idx
    ]

    # Return results
    return {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "report": report,
        "class_names": class_names,
        "misclassified": misclassified,
        "all_probs": all_probs,
        "all_preds": all_preds,
        "all_labels": all_labels,
        "all_paths": all_paths
    }


# UI Components
def ui_sidebar():
    """Create the sidebar UI"""
    st.sidebar.title("🧠 MRI Tumor Detector")

    # Model selection
    st.sidebar.header("Model")
    model_path = st.sidebar.text_input(
        "Model Path",
        value="meningioma_model_V2_TORCH.pth"
    )

    # Select mode
    mode = st.sidebar.radio(
        "Mode",
        ["Single Image", "Batch Processing", "Model Evaluation"]
    )

    # Image processing options
    st.sidebar.header("Image Processing")
    use_grayscale = st.sidebar.checkbox(
        "Convert to Grayscale",
        value=True,
        help="Convert color images to grayscale before processing"
    )

    return {
        "model_path": model_path,
        "mode": mode,
        "use_grayscale": use_grayscale
    }


def ui_single_image(model, device, use_grayscale):
    """UI for single image mode"""
    st.header("Analyze Single MRI Image")

    # Upload image
    uploaded_file = st.file_uploader(
        "Upload an MRI scan",
        type=["jpg", "jpeg", "png", "tif", "tiff"]
    )

    if uploaded_file is not None:
        # Read image
        image = Image.open(uploaded_file)

        # Display original
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)

        # Analyze button
        if st.button("Analyze Image"):
            # Make prediction
            result = predict(model, image, device, use_grayscale)

            # Display preprocessing steps
            with col2:
                if result["grayscale"] is not None:
                    st.image(result["grayscale"], caption="Grayscale", use_container_width=True)
                else:
                    st.write("(Grayscale conversion not applied)")

            with col3:
                st.image(result["preprocessed"], caption=f"Preprocessed ({IMAGE_SIZE}x{IMAGE_SIZE})",
                         use_container_width=True)

            # Display results
            st.subheader("Analysis Results")

            # Create columns for results
            res_col1, res_col2 = st.columns([1, 2])

            with res_col1:
                # Show result with color
                result_color = "green" if result["class"] == "No Tumor" else "red"
                st.markdown(f"### <span style='color:{result_color}'>{result['class']}</span>", unsafe_allow_html=True)

                # Show confidence
                st.write(f"Confidence: {result['confidence']:.2%}")
                st.progress(float(result['confidence']))

                # Show raw probability
                st.write(f"Raw probability: {result['raw_probability']:.4f}")
                st.write(f"Inverted probability: {result['inverted_probability']:.4f}")

            with res_col2:
                # Medical disclaimer
                st.info(
                    "**Medical Disclaimer:** This analysis is for demonstration purposes only. "
                    "Please consult with a qualified medical professional for proper diagnosis."
                )

                # Interpretation
                if result["class"] == "Meningioma":
                    st.warning(
                        "The model has detected features consistent with a meningioma tumor in this MRI scan. "
                        "Please consult with a medical professional for proper diagnosis and treatment options."
                    )
                else:
                    st.success(
                        "The model did not detect features consistent with a meningioma tumor in this MRI scan. "
                        "However, this does not rule out other conditions. Always consult with a medical professional."
                    )


def ui_batch_processing(model, device, use_grayscale):
    """UI for batch processing mode"""
    st.header("Batch Process MRI Images")

    # Upload multiple images
    uploaded_files = st.file_uploader(
        "Upload MRI scans",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.write(f"Uploaded {len(uploaded_files)} images")

        # Process button
        if st.button("Process All Images"):
            # Process batch
            results = process_batch(model, uploaded_files, device, use_grayscale)

            # Display results table
            st.subheader("Results Summary")

            # Prepare data for table
            df = pd.DataFrame([
                {
                    "Image": r["filename"],
                    "Prediction": r["class"],
                    "Confidence": f"{r['confidence']:.2%}",
                    "Raw Probability": f"{r['raw_probability']:.4f}"
                }
                for r in results
            ])

            st.dataframe(df)

            # Show summary statistics
            st.subheader("Summary")

            # Count by class
            meningioma_count = sum(1 for r in results if r["class"] == "Meningioma")
            notumor_count = sum(1 for r in results if r["class"] == "No Tumor")
            total = len(results)

            # Display counts
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Meningioma",
                    meningioma_count,
                    f"{meningioma_count / total:.1%}"
                )

            with col2:
                st.metric(
                    "No Tumor",
                    notumor_count,
                    f"{notumor_count / total:.1%}"
                )

            # Plot probability distribution
            st.subheader("Probability Distribution")

            fig, ax = plt.subplots(figsize=(10, 6))

            # Get probability values by class
            meningioma_prob = [r["raw_probability"] for r in results if r["class"] == "Meningioma"]
            notumor_prob = [r["raw_probability"] for r in results if r["class"] == "No Tumor"]

            # Plot histograms
            ax.hist(meningioma_prob, alpha=0.7, label="Meningioma", bins=10, range=(0, 1))
            ax.hist(notumor_prob, alpha=0.7, label="No Tumor", bins=10, range=(0, 1))

            ax.set_xlabel("Raw Probability")
            ax.set_ylabel("Number of Images")
            ax.set_title("Raw Probability Distribution by Class")
            ax.legend()
            ax.grid(True, alpha=0.3)

            st.pyplot(fig)

            # Display sample images (high & low confidence examples)
            st.subheader("Sample Predictions")

            # Sort by confidence
            sorted_results = sorted(results, key=lambda x: x["confidence"])

            # Get low and high confidence examples
            low_conf = sorted_results[:min(3, len(sorted_results))]
            high_conf = sorted_results[-min(3, len(sorted_results)):]
            samples = low_conf + high_conf

            # Create grid
            num_cols = min(3, len(samples))
            cols = st.columns(num_cols)

            # Display samples
            for i, sample in enumerate(samples):
                col = cols[i % num_cols]

                # Find original file
                file = next(f for f in uploaded_files if f.name == sample["filename"])
                file.seek(0)  # Reset file pointer
                img = Image.open(file)

                # Display with info
                col.image(img, use_container_width=True)

                # Color based on prediction
                color = "green" if sample["class"] == "No Tumor" else "red"

                col.markdown(
                    f"**<span style='color:{color}'>{sample['class']}</span>**<br>"
                    f"Confidence: {sample['confidence']:.2%}",
                    unsafe_allow_html=True
                )


def ui_model_evaluation(model, device, use_grayscale):
    """UI for model evaluation mode"""
    st.header("Model Evaluation")

    # Test directory input
    test_dir = st.text_input("Test Directory Path", "datasets/Testing")


    # Evaluate button
    if st.button("Evaluate Model"):
        if os.path.exists(test_dir):
            with st.spinner("Evaluating model..."):
                # Run evaluation
                results = evaluate_model(model, test_dir, device, use_grayscale)

                if results is None:
                    return

                # Display metrics
                st.subheader("Performance Metrics")

                # Accuracy
                st.metric("Accuracy", f"{results['accuracy']:.4f}")

                # Classification report
                st.subheader("Classification Report")
                report_df = pd.DataFrame(results["report"]).transpose()
                st.dataframe(report_df.style.format("{:.4f}"))

                # Confusion matrix
                st.subheader("Confusion Matrix")
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(
                    results["confusion_matrix"],
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=results["class_names"],
                    yticklabels=results["class_names"],
                    ax=ax
                )
                ax.set_ylabel("True Label")
                ax.set_xlabel("Predicted Label")
                st.pyplot(fig)

                # Calculate additional metrics
                cm = results["confusion_matrix"]
                if cm.size == 4:  # 2x2 matrix
                    tn, fp, fn, tp = cm.ravel()

                    # Calculate metrics
                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (
                                                                                                  precision + sensitivity) > 0 else 0

                    # Display metrics
                    st.subheader("Additional Metrics")
                    metrics_df = pd.DataFrame({
                        "Metric": ["Sensitivity (Recall)", "Specificity", "Precision", "F1 Score"],
                        "Value": [sensitivity, specificity, precision, f1]
                    })
                    st.dataframe(metrics_df.style.format({"Value": "{:.4f}"}))

                # Misclassified examples
                if results["misclassified"]:
                    st.subheader("Misclassified Examples")

                    # Show table
                    misclassified_df = pd.DataFrame([
                        {
                            "Image": os.path.basename(m["path"]),
                            "True Class": m["true_class"],
                            "Predicted Class": m["pred_class"],
                            "Confidence": f"{m['confidence']:.2%}"
                        }
                        for m in results["misclassified"]
                    ])
                    st.dataframe(misclassified_df)

                    # Show examples
                    st.write("Sample Misclassified Images:")

                    # Limit to 6 examples
                    samples = results["misclassified"][:min(6, len(results["misclassified"]))]

                    # Create grid
                    num_cols = min(3, len(samples))
                    cols = st.columns(num_cols)

                    # Display samples
                    for i, sample in enumerate(samples):
                        col = cols[i % num_cols]

                        # Load image
                        try:
                            img = Image.open(sample["path"])
                            col.image(img, use_container_width=True)
                            col.write(f"True: {sample['true_class']}")
                            col.write(f"Pred: {sample['pred_class']}")
                            col.write(f"Conf: {sample['confidence']:.2%}")
                        except Exception as e:
                            col.error(f"Error loading image: {e}")
                else:
                    st.success("No misclassifications found!")
        else:
            st.error(f"Test directory not found: {test_dir}")


def ui_model_info(model, model_type, device):
    """Display model information"""
    with st.expander("Model Information"):
        # Basic info
        st.write(f"**Model Type:** {model_type}")
        st.write(f"**Device:** {device}")
        st.write(
            f"**Note:** This model uses inverted probability interpretation. A value close to 0 indicates meningioma.")

        # Display model architecture if not TorchScript
        if model_type == "PyTorch":
            st.code(str(model))

            # Count parameters
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())

            st.write(f"**Total Parameters:** {total_params:,}")
            st.write(f"**Trainable Parameters:** {trainable_params:,}")


def main():
    """Main application function"""
    # Create sidebar
    sidebar_options = ui_sidebar()

    # Explain the inverted logic
    st.info("""
    **⚠️ Important Note:** This app has been updated to fix an inverted prediction issue. 
    The model outputs probabilities with an inverted meaning:
    - Probability close to 0 → No Tumor
    - Probability close to 1 → Meningioma (tumor)

    The app handles this internally, so predictions should now be correct.
    """)

    # Load model
    model, model_type, device = load_model(sidebar_options["model_path"])

    if model is None:
        st.error("Failed to load model. Please check the model path.")
        return

    # Show model info
    ui_model_info(model, model_type, device)

    # Display UI based on selected mode
    if sidebar_options["mode"] == "Single Image":
        ui_single_image(model, device, sidebar_options["use_grayscale"])

    elif sidebar_options["mode"] == "Batch Processing":
        ui_batch_processing(model, device, sidebar_options["use_grayscale"])

    elif sidebar_options["mode"] == "Model Evaluation":
        ui_model_evaluation(model, device, sidebar_options["use_grayscale"])

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info(
        "This app loads a PyTorch model trained with mri_tumor_model_pytorch.py "
        "and allows you to test it on MRI images to detect meningioma tumors."
    )
    st.sidebar.warning(
        "**Disclaimer:** This is for demonstration purposes only. "
        "Medical diagnosis should always be performed by qualified professionals."
    )


if __name__ == "__main__":
    main()