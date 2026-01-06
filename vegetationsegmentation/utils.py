import torch
import torchvision
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score

# -----------------------------
# Build DeepLabV3 model
# -----------------------------
def build_veg_model(num_classes=2, pretrained=False):
    """
    Build a DeepLabV3 model with a ResNet-50 backbone
    """
    model = torchvision.models.segmentation.deeplabv3_resnet50(
        pretrained=pretrained,
        progress=True
    )
    # Replace classifier with correct number of classes
    model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)
    return model

# -----------------------------
# Load model weights
# -----------------------------
def load_model(model, path=r"C:\Users\welcome\Desktop\vegtation_segmentation\VegetationSegmentation\models\vegetation_model.pth", device=None):
    """
    Load model weights from given path
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"✅ Model loaded from {path}")
    return model

# -----------------------------
# Save model helper
# -----------------------------
def save_model(model, folder="models", filename="vegetation_model.pth"):
    path = Path(folder) / filename
    path.parent.mkdir(parents=True, exist_ok=True)  # create folder if not exists
    torch.save(model.state_dict(), path)
    print(f"✅ Model saved at {path}")

# -----------------------------
# Save segmentation results
# -----------------------------
def save_segmentation_results(y_true, y_pred, folder="results", normalize_cm=False):
    """
    Save confusion matrix and F1 score for segmentation.

    Args:
        y_true (array-like or torch.Tensor): Ground truth labels
        y_pred (array-like or torch.Tensor): Predicted labels
        folder (str): Folder to save results
        normalize_cm (bool): If True, normalize confusion matrix
    """
    # Convert to numpy if torch tensors
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()

    # Create results folder
    RESULTS_FOLDER = Path(folder)
    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize_cm:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # F1 score
    f1 = f1_score(y_true, y_pred, average="macro")

    # Save files
    np.save(RESULTS_FOLDER / "confusion_matrix.npy", cm)
    with open(RESULTS_FOLDER / "f1_score.txt", "w") as f:
        f.write(f"F1 Score (macro): {f1}\n")

    # Plot confusion matrix
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize_cm else "d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(RESULTS_FOLDER / "confusion_matrix.png")
    plt.close()

    print(f"✅ Confusion matrix and F1 score saved in '{RESULTS_FOLDER}'")
