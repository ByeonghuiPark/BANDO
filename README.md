# RT-DETR Skeleton

This repository provides a minimal skeleton for developing an RT-DETR (Real-Time Detection Transformer) model.

## Structure
- `rtdetr/` – model implementation
- `datasets/` – dataset utilities
- `configs/` – YAML configuration files
- `utils/` – training helpers
- `scripts/` – example training script

## Usage
Install dependencies:
```bash
pip install torch torchvision pyyaml
```

Run training:
```bash
python scripts/train.py --config configs/default.yaml
```

This script uses a dummy dataset and runs for one epoch. Customize the dataset and configuration for real training.
