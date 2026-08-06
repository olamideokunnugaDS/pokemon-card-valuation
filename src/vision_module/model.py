"""CNN architectures for the intrinsic condition encoder.

Two-stage design:
  - Stage 1 (IdentityEncoder): learns card identity (set/artwork/era) from a
    frozen ResNet50 backbone. Its embedding is frozen after training and used
    as an orthogonality reference for Stage 2.
  - Stage 2 (ConditionEncoder): learns physical condition (grade) from the
    same frozen backbone, with a separate trainable head. An orthogonality
    penalty (OrthogonalityLoss) discourages the condition embedding from
    re-encoding identity information already captured in Stage 1.
"""

import torch
import torch.nn as nn
from torchvision import models


class IdentityEncoder(nn.Module):
    """Stage 1: card identity encoder.

    Frozen ResNet50 backbone -> trainable embedding head -> classifier.
    The embedding captures card identity (set, artwork, era) so that Stage 2
    can be trained to avoid re-encoding the same signal.
    """

    def __init__(self, embedding_dim: int = 256, num_classes: int = 7, dropout: float = 0.3):
        super().__init__()

        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # drop FC layer

        for param in self.features.parameters():
            param.requires_grad = False

        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = self.features(x)
        emb = self.embedding(features)
        logits = self.classifier(emb)
        return logits, emb

    def get_embedding(self, x):
        """Extract the identity embedding only (no classifier head)."""
        with torch.no_grad():
            features = self.features(x)
            emb = self.embedding(features)
        return emb


class ConditionEncoder(nn.Module):
    """Stage 2: condition residual encoder.

    Shares the same frozen ResNet50 backbone as Stage 1 (separate instance,
    same pretrained weights) but has its own trainable embedding head, tuned
    for grade classification under an orthogonality constraint against the
    Stage 1 identity embedding.
    """

    def __init__(self, embedding_dim: int = 256, num_classes: int = 3, dropout: float = 0.3):
        super().__init__()

        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(backbone.children())[:-1])

        for param in self.features.parameters():
            param.requires_grad = False

        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = self.features(x)
        emb = self.embedding(features)
        logits = self.classifier(emb)
        return logits, emb

    def get_embedding(self, x):
        """Extract the condition embedding only (no classifier head)."""
        with torch.no_grad():
            features = self.features(x)
            emb = self.embedding(features)
        return emb


class OrthogonalityLoss:
    """Penalises alignment between the condition and identity embeddings.

    L_ortho = mean(|cosine_similarity(emb_condition, emb_identity)|)

    The absolute value penalises both positive and negative correlation,
    since the goal is orthogonality rather than anti-correlation.
    """

    def __init__(self, identity_model: nn.Module, weight: float = 0.1):
        self.identity_model = identity_model
        self.weight = weight
        self._last_ortho_value = 0.0
        self._current_identity_emb = None

    def __call__(self, condition_emb: torch.Tensor) -> torch.Tensor:
        identity_emb = self._current_identity_emb
        cos_sim = nn.functional.cosine_similarity(condition_emb, identity_emb, dim=1)
        ortho_loss = cos_sim.abs().mean()
        self._last_ortho_value = ortho_loss.item()
        return self.weight * ortho_loss

    def set_identity_embeddings(self, identity_emb: torch.Tensor) -> None:
        """Store the identity embeddings for the current batch."""
        self._current_identity_emb = identity_emb.detach()
