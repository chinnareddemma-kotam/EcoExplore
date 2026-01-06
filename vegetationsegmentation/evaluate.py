import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import SegmentationDataset
from model import build_veg_model

# -----------------------------
# DEVICE
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# PATHS
# -----------------------------
VAL_IMAGES = r"C:\Users\welcome\Desktop\vegtation_segmentation\vegetation-segmentation-4\valid\images"
VAL_MASKS  = r"C:\Users\welcome\Desktop\vegtation_segmentation\vegetation-segmentation-4\valid\masks"

MODEL_PATH = r"C:\Users\welcome\Desktop\vegtation_segmentation\VegetationSegmentation\models\vegetation_model.pth"

# -----------------------------
# DATASET & DATALOADER
# -----------------------------
val_dataset = SegmentationDataset(VAL_IMAGES, VAL_MASKS)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)

print(f"Validation samples: {len(val_dataset)}")

# -----------------------------
# MODEL
# -----------------------------
num_classes = 2  # background + vegetation
model = build_veg_model(num_classes).to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("✅ Model loaded successfully")

# -----------------------------
# IoU FUNCTION (BINARY)
# -----------------------------
def compute_iou(outputs, masks):
    preds = torch.argmax(outputs, dim=1)

    preds = (preds == 1).float()
    masks = (masks == 1).float()

    intersection = (preds * masks).sum(dim=(1, 2))
    union = preds.sum(dim=(1, 2)) + masks.sum(dim=(1, 2)) - intersection

    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.item()
def pixel_accuracy(outputs, masks):
    preds = torch.argmax(outputs, dim=1)
    correct = (preds == masks).float().sum()
    total = masks.numel()
    return (correct / total).item()

# -----------------------------
# EVALUATION LOOP
# -----------------------------
ious = []
accuracies = []

with torch.no_grad():
    for images, masks in val_loader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)["out"]

        ious.append(compute_iou(outputs, masks))
        accuracies.append(pixel_accuracy(outputs, masks))

print("📊 Mean IoU:", round(np.mean(ious), 4))
print("📊 Pixel Accuracy:", round(np.mean(accuracies) * 100, 2), "%")
