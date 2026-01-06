from pathlib import Path
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from Soil_detection.dataset import SoilClassificationDataset
from Soil_detection.model import build_model
from Soil_detection.soil_metrics import evaluate_classification

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    project_root = Path(__file__).resolve().parents[1]
    dataset_root = project_root / "Soil-detection-3"

    with open(dataset_root / "data.yaml") as f:
        class_names = yaml.safe_load(f)["names"]

    train_ds = SoilClassificationDataset(
        dataset_root / "train/images",
        dataset_root / "train/labels",
        class_names,
        augment=True,
    )

    val_ds = SoilClassificationDataset(
        dataset_root / "valid/images",
        dataset_root / "valid/labels",
        class_names,
    )

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)

    model = build_model("resnet50", len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)

    best_acc = 0
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)

    for epoch in range(1, 21):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()

        val_metrics = evaluate_classification(model, val_loader, criterion, device)
        print(f"Epoch {epoch} | Val Acc: {val_metrics['accuracy']:.4f}")

        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            torch.save({
                "model": model.state_dict(),
                "class_names": class_names
            }, models_dir / "best_soil_classifier.pth")

if __name__ == "__main__":
    main()
