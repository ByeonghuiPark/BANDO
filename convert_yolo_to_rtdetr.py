"""Convert a YOLO formatted dataset to a COCO style dataset for RT-DETR."""

import argparse
import json
import os
from pathlib import Path
from typing import List

from PIL import Image


def load_classes(classes_file: Path) -> List[str]:
    """Load class names from a text file."""
    with open(classes_file, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def convert_split(image_dir: Path, label_dir: Path, classes: List[str], output_json: Path) -> None:
    images = []
    annotations = []
    ann_id = 1
    image_id = 1

    for img_name in sorted(os.listdir(image_dir)):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue
        img_path = image_dir / img_name
        with Image.open(img_path) as img:
            width, height = img.size

        images.append({
            "id": image_id,
            "file_name": str(img_path),
            "width": width,
            "height": height,
        })

        label_path = label_dir / f"{Path(img_name).stem}.txt"
        if label_path.exists():
            with open(label_path, 'r', encoding='utf-8') as lf:
                for line in lf:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls, x_center, y_center, w, h = map(float, parts)
                    x_min = (x_center - w / 2) * width
                    y_min = (y_center - h / 2) * height
                    bbox_w = w * width
                    bbox_h = h * height

                    annotations.append({
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": int(cls),
                        "bbox": [x_min, y_min, bbox_w, bbox_h],
                        "area": bbox_w * bbox_h,
                        "iscrowd": 0,
                    })
                    ann_id += 1
        image_id += 1

    categories = [{"id": idx, "name": name} for idx, name in enumerate(classes)]
    data = {"images": images, "annotations": annotations, "categories": categories}
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(data, f)


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO to RT-DETR dataset converter")
    parser.add_argument('--dataset', type=Path, required=True, help='Root directory of YOLO dataset')
    parser.add_argument('--classes', type=Path, required=True, help='Text file with class names, one per line')
    parser.add_argument('--output', type=Path, required=True, help='Directory to save converted COCO JSON files')
    args = parser.parse_args()

    classes = load_classes(args.classes)
    args.output.mkdir(parents=True, exist_ok=True)

    for split in ['train', 'val', 'test']:
        img_dir = args.dataset / 'images' / split
        lbl_dir = args.dataset / 'labels' / split
        if img_dir.exists() and lbl_dir.exists():
            out_json = args.output / f'{split}.json'
            convert_split(img_dir, lbl_dir, classes, out_json)
            print(f'Converted {split} dataset -> {out_json}')


if __name__ == '__main__':
    main()
