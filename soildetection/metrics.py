import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

def evaluate_classification(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    num_classes = loader.dataset.num_classes
    conf_mat = np.zeros((num_classes, num_classes), dtype=int)

    with torch.no_grad():
        for imgs, labels in tqdm(loader, leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for t, p in zip(labels.cpu(), preds.cpu()):
                conf_mat[t, p] += 1

    per_class = []
    for i in range(num_classes):
        tp = conf_mat[i, i]
        fp = conf_mat[:, i].sum() - tp
        fn = conf_mat[i].sum() - tp
        per_class.append({
            "precision": tp / (tp + fp + 1e-7),
            "recall": tp / (tp + fn + 1e-7),
            "support": int(conf_mat[i].sum())
        })

    return {
        "loss": total_loss / len(loader),
        "accuracy": correct / total,
        "confusion_matrix": conf_mat,
        "per_class": per_class
    }

def plot_confusion_matrix(cm, class_names, out_path):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(len(class_names)), class_names, rotation=45)
    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
