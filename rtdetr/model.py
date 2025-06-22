import torch
from torch import nn

class RTDETR(nn.Module):
    """Simplified Real-Time Detection Transformer."""
    def __init__(self, num_classes: int, num_queries: int = 100):
        super().__init__()
        self.backbone = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.transformer = nn.Transformer(d_model=64, nhead=8, num_encoder_layers=6,
                                          num_decoder_layers=6)
        self.class_embed = nn.Linear(64, num_classes)
        self.bbox_embed = nn.Linear(64, 4)
        self.query_embed = nn.Embedding(num_queries, 64)

    def forward(self, images: torch.Tensor):
        # images: (batch_size, 3, H, W)
        features = self.backbone(images)  # (batch_size, 64, H/2, W/2)
        bs, c, h, w = features.shape
        features = features.flatten(2).permute(2, 0, 1)  # (hw, bs, c)
        queries = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        hs = self.transformer(src=features, tgt=queries)
        outputs_class = self.class_embed(hs)
        outputs_bbox = self.bbox_embed(hs).sigmoid()
        return {"pred_logits": outputs_class[-1], "pred_boxes": outputs_bbox[-1]}
