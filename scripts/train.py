import argparse
import yaml
import torch

from rtdetr.model import RTDETR
from datasets.detection_dataset import DetectionDataset
from utils.train_utils import train_one_epoch, create_dataloader


def load_config(path: str):
    with open(path) as f:
        return yaml.safe_load(f)


def main(config_path: str):
    cfg = load_config(config_path)
    model_cfg = cfg['model']
    train_cfg = cfg['train']

    model = RTDETR(num_classes=model_cfg['num_classes'],
                   num_queries=model_cfg['num_queries'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # dummy dataset with a single image and annotation
    annotations = [([0.1, 0.1, 0.2, 0.2], 1)]
    dataset = DetectionDataset('data/images', annotations)
    dataloader = create_dataloader(dataset, train_cfg['batch_size'])

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg['lr'])

    for epoch in range(train_cfg['epochs']):
        train_one_epoch(model, dataloader, optimizer, device)
        print(f"Epoch {epoch+1} completed")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    args = parser.parse_args()
    main(args.config)
