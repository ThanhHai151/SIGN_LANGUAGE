"""
WordRecognizer716: SSTCN_Attention adapted for 716-class Vietnamese Sign Language Vocabulary.

Input: [B, 3, 30, 75]
  - Channels: 3 (X, Y, Z coordinates image-normalized)
  - Time: 30 frames
  - Joints: 75 landmarks (0..32 Pose, 33..53 Left Hand, 54..74 Right Hand)
Output: [B, 716] logits

Features:
- Exact 29 backbone tensors transfer from pretrained SSTCN.
- Proper BatchNorm freezing (keeps running_mean and running_var static when backbone is frozen).
- Active class masking: supports both full 703 active classes and Pilot 40 active classes.
"""
import hashlib
import os
from typing import Dict, List, Optional, Tuple, Set

import torch
import torch.nn as nn
import torch.nn.functional as F

POSE_START, POSE_END = 0, 33
L_HAND_START, L_HAND_END = 33, 54
R_HAND_START, R_HAND_END = 54, 75


class JointAttention(nn.Module):
    def __init__(self, in_channels: int = 128, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B*T, C, J] -> [B*T, J, C] for MultiheadAttention
        x = x.permute(0, 2, 1)
        x, _ = self.attn(x, x, x)
        x = x.permute(0, 2, 1)  # [B*T, C, J]
        return x


class WordRecognizer716(nn.Module):
    def __init__(
        self,
        num_classes: int = 716,
        num_joints: int = 75,
        in_channels: int = 3,
        dropout: float = 0.3,
        active_indices: Optional[List[int]] = None,
        unavailable_indices: Optional[List[int]] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_joints = num_joints
        self.in_channels = in_channels
        self.dropout = dropout
        self.freeze_mode = "none"  # "none", "full", "partial"

        # --- Spatial Convolution ---
        self.spatial_conv = nn.Conv1d(in_channels, 128, kernel_size=1)
        self.spatial_bn = nn.BatchNorm1d(128)
        self.spatial_drop = nn.Dropout(dropout)

        # --- Multi-head Joint Attention for hands ---
        self.left_hand_attn = JointAttention(128, num_heads=4)
        self.right_hand_attn = JointAttention(128, num_heads=4)

        # --- Temporal Convolutions (scanning over time) ---
        self.temporal_conv1 = nn.Conv2d(128, 256, kernel_size=(3, 1), padding=(1, 0))
        self.temporal_bn1 = nn.BatchNorm2d(256)
        self.temporal_drop1 = nn.Dropout2d(dropout)

        self.temporal_conv2 = nn.Conv2d(256, 512, kernel_size=(3, 1), padding=(1, 0))
        self.temporal_bn2 = nn.BatchNorm2d(512)
        self.temporal_drop2 = nn.Dropout2d(dropout)

        # --- 716-Class Linear Classifier Head ---
        self.fc_drop = nn.Dropout(dropout)
        self.fc = nn.Linear(512, num_classes)

        # Joint index slices
        self.left_hand_joints = list(range(L_HAND_START, L_HAND_END))
        self.right_hand_joints = list(range(R_HAND_START, R_HAND_END))

        # Register active / available classes mask
        mask = torch.zeros(num_classes, dtype=torch.bool)
        if active_indices is not None:
            for idx in active_indices:
                if 0 <= idx < num_classes:
                    mask[idx] = True
            self.active_indices = sorted(active_indices)
            self.unavailable_indices = [i for i in range(num_classes) if i not in set(self.active_indices)]
        else:
            mask = torch.ones(num_classes, dtype=torch.bool)
            unavail = set(unavailable_indices or [])
            for idx in unavail:
                if 0 <= idx < num_classes:
                    mask[idx] = False
            self.unavailable_indices = sorted(unavail)
            self.active_indices = [i for i in range(num_classes) if i not in unavail]

        self.register_buffer("available_classes_mask", mask)

    def train(self, mode: bool = True):
        """
        Override train() to ensure frozen BatchNorm layers do NOT update
        running_mean and running_var when backbone is frozen.
        """
        super().train(mode)
        if mode and self.freeze_mode == "full":
            self.spatial_bn.eval()
            self.temporal_bn1.eval()
            self.temporal_bn2.eval()
        elif mode and self.freeze_mode == "partial":
            self.spatial_bn.eval()
            self.temporal_bn1.eval()
            self.temporal_bn2.train()
        return self

    def forward(self, x: torch.Tensor, apply_class_mask: bool = False) -> torch.Tensor:
        # Input shape: [B, C, T, J] = [B, 3, 30, 75]
        B, C, T, J = x.shape

        # 1. Spatial conv per frame
        x = x.permute(0, 2, 1, 3).contiguous()  # [B, T, C, J]
        x = x.view(B * T, C, J)                 # [B*T, C, J]
        x = F.relu(self.spatial_bn(self.spatial_conv(x)))
        x = self.spatial_drop(x)

        # 2. Joint Attention for left and right hand landmarks
        left_hand = x[:, :, self.left_hand_joints]
        left_hand = self.left_hand_attn(left_hand)
        right_hand = x[:, :, self.right_hand_joints]
        right_hand = self.right_hand_attn(right_hand)

        # Update hand features back into frame tensor
        x_new = x.clone()
        lh_idx = torch.tensor(self.left_hand_joints, device=x.device).view(1, 1, -1).expand(B * T, 128, -1)
        rh_idx = torch.tensor(self.right_hand_joints, device=x.device).view(1, 1, -1).expand(B * T, 128, -1)
        x_new.scatter_(2, lh_idx, left_hand)
        x_new.scatter_(2, rh_idx, right_hand)
        x = x_new

        # 3. Reshape for Temporal Convolution: [B, 128, T, J]
        x = x.view(B, T, 128, J).permute(0, 2, 1, 3)

        # 4. Temporal Convolutions
        x = F.relu(self.temporal_bn1(self.temporal_conv1(x)))
        x = self.temporal_drop1(x)
        x = F.relu(self.temporal_bn2(self.temporal_conv2(x)))
        x = self.temporal_drop2(x)

        # 5. Global Average Pooling over time and joints -> [B, 512]
        x = x.mean(dim=2).mean(dim=2)

        # 6. Classifier Head -> [B, 716]
        x = self.fc_drop(x)
        logits = self.fc(x)

        if apply_class_mask:
            logits = self.apply_mask(logits)

        return logits

    def apply_mask(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Mask out inactive classes (setting their logits to -50.0)
        so their probability after softmax is virtually zero (exp(-50) ~ 1.9e-22).
        """
        mask = self.available_classes_mask.to(logits.device)
        masked_logits = logits.clone()
        masked_logits[:, ~mask] = -50.0
        return masked_logits

    def load_pretrained_backbone(self, weights_path: str) -> Dict[str, any]:
        """
        Load backbone weights from pretrained SSTCN checkpoint.
        Transfers 29 backbone tensors, ignoring the old 102-class fc layers.
        """
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

        h = hashlib.sha256()
        with open(weights_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        sha256_hash = h.hexdigest()

        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        src_sd = ckpt.get("model_state_dict", ckpt)

        current_sd = self.state_dict()
        loaded_keys = []
        skipped_keys = []

        for key, val in src_sd.items():
            if key in current_sd:
                if current_sd[key].shape == val.shape:
                    current_sd[key].copy_(val)
                    loaded_keys.append(key)
                else:
                    skipped_keys.append((key, f"shape_mismatch:{val.shape}_vs_{current_sd[key].shape}"))
            else:
                skipped_keys.append((key, "key_not_in_target"))

        self.load_state_dict(current_sd)
        return {
            "source_weights_path": os.path.abspath(weights_path),
            "source_sha256": sha256_hash,
            "loaded_backbone_keys_count": len(loaded_keys),
            "loaded_keys": loaded_keys,
            "skipped_keys": skipped_keys,
        }

    def freeze_backbone(self, freeze: bool = True):
        """
        Phase A: Freeze all backbone layers, leaving only the 716-class fc head trainable.
        Explicitly sets BatchNorm layers to eval() mode to freeze running stats.
        """
        self.freeze_mode = "full" if freeze else "none"
        for name, param in self.named_parameters():
            if "fc" not in name:
                param.requires_grad = not freeze
            else:
                param.requires_grad = True

        if freeze:
            self.spatial_bn.eval()
            self.temporal_bn1.eval()
            self.temporal_bn2.eval()

    def unfreeze_partial_backbone(self, unfreeze_temporal2: bool = True):
        """
        Phase B: Unfreeze temporal_conv2 / temporal_bn2 + fc for partial fine-tuning.
        Early layers (spatial_conv, hand attention, temporal_conv1) remain frozen.
        """
        self.freeze_mode = "partial" if unfreeze_temporal2 else "full"
        for name, param in self.named_parameters():
            if "fc" in name:
                param.requires_grad = True
            elif unfreeze_temporal2 and ("temporal_conv2" in name or "temporal_bn2" in name):
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.spatial_bn.eval()
        self.temporal_bn1.eval()
        if unfreeze_temporal2:
            self.temporal_bn2.train()
        else:
            self.temporal_bn2.eval()

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return {"total": total, "trainable": trainable, "frozen": frozen}
