"""Grad-CAM and shortcut-detection utilities for the condition encoder."""

from typing import Callable, Tuple

import numpy as np
import torch


def gradcam_forward(model, x):
    """Forward pass that exposes embeddings for gradient flow.

    Bind to a model instance to enable Grad-CAM on models whose default
    `forward` was written for inference only, e.g.:
        model.forward = lambda x: gradcam_forward(model, x)
    """
    features = model.features(x)
    emb = model.embedding(features)
    logits = model.classifier(emb)
    return logits, emb


class GradCAM:
    """Grad-CAM for a ResNet50-based encoder."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        prev_requires_grad = {}
        for name, param in self.model.named_parameters():
            prev_requires_grad[name] = param.requires_grad
            param.requires_grad = True

        self.model.eval()
        input_tensor.requires_grad = True

        output, _ = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot)

        for name, param in self.model.named_parameters():
            param.requires_grad = prev_requires_grad[name]

        if self.gradients is None:
            return np.zeros((7, 7))  # fallback if no gradient was captured

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.squeeze().cpu().numpy()


def create_body_masked_transform(base_transform: Callable, mask_region: Tuple[float, float] = (0.15, 0.90)):
    """Build a transform that zeroes out the card-body region of an image.

    Used as a shortcut-learning probe: if masking the card body collapses
    model accuracy toward the majority-class baseline, the model is relying
    on genuine body content rather than the PSA label or slab casing.
    """

    class BodyMaskTransform:
        def __call__(self, img):
            tensor = base_transform(img)
            h = tensor.shape[1]
            start = int(h * mask_region[0])
            end = int(h * mask_region[1])
            tensor[:, start:end, :] = 0
            return tensor

    return BodyMaskTransform()
