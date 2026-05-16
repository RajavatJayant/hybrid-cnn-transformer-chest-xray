import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from config import CONFIG


class HybridModel(nn.Module):
    """
    DenseNet-121 CNN backbone + Transformer encoder head for
    multi-label chest X-ray classification.

    Architecture
    ------------
    1. DenseNet-121 extracts spatial features  → (B, 1024, 7, 7)
    2. Global Average Pooling                  → (B, 1024)
    3. Linear projection                       → (B, 512)
    4. 2-layer Transformer encoder             → (B, 512)
    5. Linear classifier                       → (B, NUM_CLASSES)
    """

    def __init__(self, num_classes: int = CONFIG['NUM_CLASSES']):
        super().__init__()

        # ── Backbone ──────────────────────────────────────────────
        backbone = models.densenet121(weights='DEFAULT')
        num_features = backbone.classifier.in_features
        backbone.classifier = nn.Identity()          # Remove original head
        self.backbone = backbone

        # ── Projection + Transformer ──────────────────────────────
        self.projection = nn.Linear(num_features, 512)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=512, nhead=8, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # ── Classifier ────────────────────────────────────────────
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN features via DenseNet (GAP applied inside backbone with Identity head)
        features = self.backbone(x)                  # (B, 1024)

        # Project → add sequence dim → Transformer → remove sequence dim
        proj  = self.projection(features).unsqueeze(1)   # (B, 1, 512)
        trans = self.transformer(proj).squeeze(1)         # (B, 512)

        return self.classifier(trans)                     # (B, NUM_CLASSES)
