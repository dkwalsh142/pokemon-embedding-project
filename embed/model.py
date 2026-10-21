"""Stage 1 embedding model: ResNet-50 backbone + linear embedding layer, plus an ArcFace head.

The ArcFace head is only used during training. Afterwards it is thrown away and
CardEmbedder alone (backbone + embedding layer) is the embedding model.

Smoke test:
    python -m embed.model
"""

import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class CardEmbedder(nn.Module):
    """Image -> 256-d embedding. Use .embed() to get L2-normalized vectors."""

    def __init__(self, emb_dim=256, dropout=0.2, pretrained=True):
        super().__init__()
        # num_classes=0 removes ResNet's ImageNet classifier; output is the
        # 2048-d global-average-pooled feature vector.
        self.backbone = timm.create_model("resnet50", pretrained=pretrained, num_classes=0)
        self.dropout = nn.Dropout(dropout)
        # No activation after this layer: a ReLU would zero every negative
        # value and distort cosine similarity.
        self.embedding = nn.Linear(self.backbone.num_features, emb_dim)

    def forward(self, x):
        return self.embedding(self.dropout(self.backbone(x)))

    def embed(self, x):
        return F.normalize(self.forward(x), dim=1)

    def freeze_backbone(self, frozen):
        for p in self.backbone.parameters():
            p.requires_grad = not frozen


class ArcFaceHead(nn.Module):
    """Cosine classifier with an additive angular margin (Deng et al., 2019).

    Each class has a learned weight vector (its "centre"). The logit for a class
    is the cosine between the embedding and that centre. For the correct class we
    add a margin m to the angle before taking the cosine, so the model must pull
    an embedding at least m radians closer to its own centre than to any other.
    """

    def __init__(self, emb_dim, n_classes, s=30.0, m=0.3):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_classes, emb_dim))
        nn.init.xavier_uniform_(self.weight)
        self.s = s  # scale: cosines are in [-1, 1], too small for softmax on their own
        self.m = m  # angular margin in radians
        self.cos_m, self.sin_m = math.cos(m), math.sin(m)
        # Past this point theta + m would exceed pi and cos(theta + m) would start
        # increasing again, so we switch to a linear penalty instead.
        self.threshold = math.cos(math.pi - m)
        self.fallback = math.sin(math.pi - m) * m

    def forward(self, emb, labels=None):
        # Cosine similarity between every embedding and every class centre.
        cos = F.linear(F.normalize(emb.float()), F.normalize(self.weight))
        if labels is None:
            return cos * self.s  # plain scaled cosine, used for accuracy

        # cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        sin = torch.sqrt((1.0 - cos ** 2).clamp(min=1e-7))
        cos_margin = cos * self.cos_m - sin * self.sin_m
        cos_margin = torch.where(cos > self.threshold, cos_margin, cos - self.fallback)

        # Apply the margin to the correct class only.
        is_target = F.one_hot(labels, cos.shape[1]).bool()
        logits = torch.where(is_target, cos_margin, cos)
        return logits * self.s


if __name__ == "__main__":
    model = CardEmbedder(pretrained=False)
    head = ArcFaceHead(256, n_classes=10)
    x = torch.randn(4, 3, 320, 224)  # batch of 4 cards, 224 wide x 320 tall
    labels = torch.tensor([0, 3, 3, 9])

    emb = model.embed(x)
    logits = head(model(x), labels)
    loss = F.cross_entropy(logits, labels)
    print("embedding:", tuple(emb.shape), "norms:", emb.norm(dim=1).tolist())
    print("logits:", tuple(logits.shape), "loss:", round(loss.item(), 3))
    print("parameters: %.1fM" % (sum(p.numel() for p in model.parameters()) / 1e6))
