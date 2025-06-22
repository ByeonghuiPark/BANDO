import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CocoDetection

# Placeholder for RT-DETR model. In practice you should import
# an implementation from an RT-DETR library or define the model here.
class RTDETRModel(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        # TODO: replace this with actual RT-DETR implementation
        self.backbone = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.classifier = nn.Linear(16, num_classes)

    def forward(self, images):
        # Simplified forward pass
        x = self.backbone(images)
        x = x.mean([2, 3])
        logits = self.classifier(x)
        return logits


def parse_args():
    parser = argparse.ArgumentParser(description="RT-DETR Training Script")
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to COCO dataset root')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    return parser.parse_args()


def main():
    args = parse_args()

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # COCO dataset
    train_dataset = CocoDetection(
        root=f"{args.data_path}/train2017",
        annFile=f"{args.data_path}/annotations/instances_train2017.json",
        transform=transform,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=lambda x: tuple(zip(*x)))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RTDETRModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            images = torch.stack(images).to(device)
            # For demonstration, this example uses the first object's label
            labels = torch.tensor([t[0]['category_id'] for t in targets]).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {running_loss/len(train_loader):.4f}")

    print('Training complete')


if __name__ == '__main__':
    main()

