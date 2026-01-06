from pathlib import Path
from typing import List, Tuple
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class SoilClassificationDataset(Dataset):
    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
        class_names: List[str],
        image_size: int = 224,
        augment: bool = False,
    ):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.class_names = class_names
        self.num_classes = len(class_names)

        self.samples: List[Tuple[Path, int]] = []

        for img_path in sorted(images_dir.glob("*")):
            label_path = labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue

            with label_path.open() as f:
                line = f.readline().strip()
                if not line:
                    continue
                class_id = int(line.split()[0])

            self.samples.append((img_path, class_id))

        aug = []
        if augment:
            aug = [
                T.RandomHorizontalFlip(),
                T.RandomRotation(15),
                T.ColorJitter(0.2, 0.2, 0.2),
            ]

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            *aug,
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img), torch.tensor(label, dtype=torch.long)
