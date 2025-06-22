import torch
from torch.utils.data import DataLoader


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    for images, targets in dataloader:
        images = images.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}
        outputs = model(images)
        loss_cls = torch.nn.functional.cross_entropy(outputs['pred_logits'], targets['labels'])
        loss_bbox = torch.nn.functional.l1_loss(outputs['pred_boxes'], targets['boxes'])
        loss = loss_cls + loss_bbox
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def create_dataloader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
