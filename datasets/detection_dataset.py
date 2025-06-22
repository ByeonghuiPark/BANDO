from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torchvision import transforms

class DetectionDataset(torch.utils.data.Dataset):
    """A simple dataset for object detection."""
    def __init__(self, image_dir: str, annotations: List[Tuple[List[float], int]],
                 transform=None):
        self.image_dir = Path(image_dir)
        self.annotations = annotations
        self.transform = transform or transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        boxes, label = self.annotations[idx]
        img_path = self.image_dir / f"{idx}.jpg"
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        target = {
            "boxes": torch.tensor([boxes], dtype=torch.float32),
            "labels": torch.tensor([label], dtype=torch.int64)
        }
        return image, target
