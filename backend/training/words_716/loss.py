"""
Loss functions for Words 716 Fine-Tuning.
Implements MaskedCrossEntropyLoss with Active-Class Label Smoothing.

Contract:
- Input logits: [B, 716]
- Targets: [B] containing original class_index in 0..715
- Label smoothing: distributes smoothing probability strictly over the active classes
  participating in training (e.g., 40 classes in Pilot or 703 in full),
  preventing probability leakage to inactive/masked classes.
"""
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        num_classes: int = 716,
        active_indices: Optional[List[int]] = None,
        unavailable_indices: Optional[List[int]] = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing

        if active_indices is not None:
            active = sorted(set(active_indices))
        elif unavailable_indices is not None:
            unavail = set(unavailable_indices)
            active = [i for i in range(num_classes) if i not in unavail]
        else:
            active = list(range(num_classes))

        self.num_active = len(active)
        self.register_buffer("active_indices_tensor", torch.tensor(active, dtype=torch.long))

        # Boolean mask over 716 classes
        mask = torch.zeros(num_classes, dtype=torch.bool)
        mask[self.active_indices_tensor] = True
        self.register_buffer("available_classes_mask", mask)

        # Mapping from original class_index (0..715) to active sub-index (0..K-1)
        target_to_sub = torch.full((num_classes,), -1, dtype=torch.long)
        for sub_idx, orig_idx in enumerate(active):
            target_to_sub[orig_idx] = sub_idx
        self.register_buffer("target_to_sub", target_to_sub)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [B, 716], targets: [B] in range [0..715]
        device = logits.device
        active_idx = self.active_indices_tensor.to(device)
        target_map = self.target_to_sub.to(device)

        # 1. Map targets to 0..K-1
        sub_targets = target_map[targets]
        if torch.any(sub_targets < 0):
            invalid_indices = targets[sub_targets < 0].tolist()
            raise ValueError(f"Targets contain inactive/unavailable class indices: {invalid_indices}")

        # 2. Extract logits strictly for active classes: [B, K]
        active_logits = logits[:, active_idx]

        # 3. Compute Cross Entropy with label smoothing confined strictly to active classes
        loss = F.cross_entropy(
            active_logits,
            sub_targets,
            label_smoothing=self.label_smoothing,
        )
        return loss
