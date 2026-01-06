from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch


class SegmentationDataset(Dataset):
    """
    Dataset class for vegetation segmentation.
    Expects:
      - image_dir: folder containing input images
      - mask_dir: folder containing corresponding masks
    """

    def __init__(self, image_dir, mask_dir):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        # Collect image-mask pairs
        self.images = []
        self.masks = []

        for img_path in sorted(self.image_dir.glob("*.jpg")):
            mask_path = self.mask_dir / img_path.name
            if mask_path.exists():
                self.images.append(img_path)
                self.masks.append(mask_path)

        print(f"Loaded {len(self.images)} image-mask pairs")

        # Image transform
        self.image_transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Mask transform (NO normalization, nearest interpolation)
        self.mask_transform = T.Compose([
            T.Resize((256, 256), interpolation=Image.NEAREST),
            T.PILToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image and mask
        image = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx])

        # Apply transforms
        image = self.image_transform(image)
        mask = self.mask_transform(mask)

        # Binary segmentation: 1 for vegetation, 0 for background
        mask = mask.squeeze(0)
        mask = (mask > 0).long()

        return image, mask


# -----------------------------
# Helper function for inference
# -----------------------------
def get_transforms():
    """
    Returns image and mask transforms for vegetation segmentation
    Can be used for standalone inference scripts
    """
    image_transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    mask_transform = T.Compose([
        T.Resize((256, 256), interpolation=Image.NEAREST),
        T.PILToTensor()
    ])

    return image_transform, mask_transform
