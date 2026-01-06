from pathlib import Path
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# Now imports will work
from dataset import SoilClassificationDataset
from model import build_model
from metrics import evaluate_classification, plot_confusion_matrix


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    project_root = Path(__file__).resolve().parents[1]
    dataset_root = project_root / "Soil-detection-3"

    with open(dataset_root / "data.yaml") as f:
        class_names = yaml.safe_load(f)["names"]

    test_ds = SoilClassificationDataset(
        dataset_root / "test/images",
        dataset_root / "test/labels",
        class_names,
    )

    loader = DataLoader(test_ds, batch_size=16)

    ckpt = torch.load(Path(__file__).parent / "models/best_soil_classifier.pth")
    model = build_model("resnet50", len(class_names), pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])

    metrics = evaluate_classification(model, loader, nn.CrossEntropyLoss(), device)

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    plot_confusion_matrix(
        metrics["confusion_matrix"],
        class_names,
        results_dir / "confusion_matrix.png"
    )

    print("Test Accuracy:", metrics["accuracy"])

if __name__ == "__main__":
    main()
