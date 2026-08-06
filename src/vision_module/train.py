"""Training and evaluation loops for the vision module.

Stage 1 (identity) uses the plain `train_one_epoch` / `evaluate` pair.
Stage 2 (condition) uses the orthogonality-aware `train_one_epoch_s2` /
`evaluate_s2` pair, which also forwards a frozen Stage 1 model to compute
the identity-vs-condition cosine similarity used by `OrthogonalityLoss`.
"""

from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn


def train_one_epoch(model, loader, criterion, optimizer, device, extra_loss_fn: Optional[Callable] = None):
    """Train for one epoch.

    Args:
        extra_loss_fn: optional callable(embeddings) -> loss tensor, used for
            the Stage 2 orthogonality penalty when training single-stage.
    """
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_extra_loss = 0.0
    correct = 0
    total = 0

    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits, emb = model(images)

        cls_loss = criterion(logits, labels)
        loss = cls_loss

        if extra_loss_fn is not None:
            extra = extra_loss_fn(emb)
            loss = loss + extra
            total_extra_loss += extra.item() * images.size(0)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_cls_loss += cls_loss.item() * images.size(0)
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return {
        "loss": total_loss / total,
        "cls_loss": total_cls_loss / total,
        "extra_loss": total_extra_loss / total if extra_loss_fn else 0.0,
        "accuracy": correct / total * 100,
    }


def evaluate(model, loader, criterion, device, extra_loss_fn: Optional[Callable] = None):
    """Evaluate a model on a held-out loader."""
    model.eval()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_extra_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            logits, emb = model(images)

            cls_loss = criterion(logits, labels)
            loss = cls_loss

            if extra_loss_fn is not None:
                extra = extra_loss_fn(emb)
                loss = loss + extra
                total_extra_loss += extra.item() * images.size(0)

            total_loss += loss.item() * images.size(0)
            total_cls_loss += cls_loss.item() * images.size(0)
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return {
        "loss": total_loss / total,
        "cls_loss": total_cls_loss / total,
        "extra_loss": total_extra_loss / total if extra_loss_fn else 0.0,
        "accuracy": correct / total * 100,
        "predictions": np.array(all_preds),
        "labels": np.array(all_labels),
    }


def train_one_epoch_s2(stage2_model, stage1_model, loader, criterion, optimizer, ortho_loss_fn, device):
    """Train Stage 2 for one epoch under the orthogonality constraint."""
    stage2_model.train()
    stage1_model.eval()  # Stage 1 is always frozen at this point

    total_loss = 0.0
    total_cls_loss = 0.0
    total_ortho_loss = 0.0
    correct = 0
    total = 0
    cos_sims = []

    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        logits, cond_emb = stage2_model(images)

        with torch.no_grad():
            _, identity_emb = stage1_model(images)

        cls_loss = criterion(logits, labels)

        cos_sim = nn.functional.cosine_similarity(cond_emb, identity_emb, dim=1)
        ortho_loss = cos_sim.abs().mean()
        weighted_ortho = ortho_loss_fn.weight * ortho_loss

        loss = cls_loss + weighted_ortho

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_cls_loss += cls_loss.item() * batch_size
        total_ortho_loss += weighted_ortho.item() * batch_size
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += batch_size
        cos_sims.append(cos_sim.detach().cpu())

    all_cos = torch.cat(cos_sims)
    return {
        "loss": total_loss / total,
        "cls_loss": total_cls_loss / total,
        "ortho_loss": total_ortho_loss / total,
        "accuracy": correct / total * 100,
        "mean_cos_sim": all_cos.abs().mean().item(),
    }


def evaluate_s2(stage2_model, stage1_model, loader, criterion, ortho_loss_fn, device):
    """Evaluate Stage 2 including orthogonality diagnostics."""
    stage2_model.eval()
    stage1_model.eval()

    total_loss = 0.0
    total_cls_loss = 0.0
    total_ortho_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    cos_sims = []

    with torch.no_grad():
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)

            logits, cond_emb = stage2_model(images)
            _, identity_emb = stage1_model(images)

            cls_loss = criterion(logits, labels)
            cos_sim = nn.functional.cosine_similarity(cond_emb, identity_emb, dim=1)
            ortho_loss = cos_sim.abs().mean()
            weighted_ortho = ortho_loss_fn.weight * ortho_loss
            loss = cls_loss + weighted_ortho

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_cls_loss += cls_loss.item() * batch_size
            total_ortho_loss += weighted_ortho.item() * batch_size
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += batch_size
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            cos_sims.append(cos_sim.cpu())

    all_cos = torch.cat(cos_sims)
    return {
        "loss": total_loss / total,
        "cls_loss": total_cls_loss / total,
        "ortho_loss": total_ortho_loss / total,
        "accuracy": correct / total * 100,
        "mean_cos_sim": all_cos.abs().mean().item(),
        "predictions": np.array(all_preds),
        "labels": np.array(all_labels),
    }
