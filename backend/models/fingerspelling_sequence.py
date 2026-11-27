"""Compact temporal classifier for 21 observed hand landmarks."""

import numpy as np
import torch
from torch import nn

SEQUENCE_FRAMES = 24
FEATURE_CHANNELS = 72


def select_dominant_hand(sequences):
    """Select a hand from [N,3,T,75] source skeletons and reflect left hands."""
    frames = np.asarray(sequences).transpose(0, 2, 3, 1)
    n, t, _, _ = frames.shape
    hands = frames[:, :, 33:].reshape(n, t, 2, 21, 3)
    visible = np.linalg.norm(hands[:, :, :, 0] - hands[:, :, :, 9], axis=-1) > 0.002
    wrist_y = hands[:, :, :, 0, 1]
    count = visible.sum(axis=1)
    mean_y = np.where(visible, wrist_y, 0).sum(axis=1) / np.maximum(count, 1)
    side = (count - 0.6 * t * mean_y).argmax(axis=1)
    selected = hands[np.arange(n), :, side].copy()
    selected[..., 0] = np.where((side == 0)[:, None, None], 1 - selected[..., 0], selected[..., 0])
    selected[~visible[np.arange(n), :, side]] = 0
    return selected


def sequence_features(hands):
    """Normalize orientation-sensitive hand shape and wrist motion separately."""
    hands = np.asarray(hands, dtype=np.float32)
    output = np.zeros((len(hands), FEATURE_CHANNELS, SEQUENCE_FRAMES), dtype=np.float32)
    for out, clip in zip(output, hands):
        visible = np.flatnonzero(np.linalg.norm(clip[:, 0] - clip[:, 9], axis=1) > 0.002)
        if len(visible) < 4:
            continue
        # Trim leading/trailing absent frames and interpolate within recorded bounds.
        sample_at = np.linspace(visible[0], visible[-1], SEQUENCE_FRAMES)
        right = np.clip(np.searchsorted(visible, sample_at, side="right"), 1, len(visible) - 1)
        low, high = visible[right - 1], visible[right]
        weight = ((sample_at - low) / (high - low)).astype(np.float32)
        sample = clip[low] + (clip[high] - clip[low]) * weight[:, None, None]
        wrist = sample[:, 0]
        shifted = sample - wrist[:, None]
        scale = np.maximum(np.linalg.norm(shifted[:, 9], axis=1), 0.008)
        shape = np.clip(shifted / scale[:, None, None], -4, 4)
        # Palm scale is consistent throughout a window for the wrist trajectory.
        path = np.clip((wrist - np.median(wrist, axis=0)) / np.median(scale), -5, 5)
        delta = np.diff(path, axis=0, prepend=path[:1])
        tips = shape[:, [4, 8, 12, 16, 20]]
        gaps = np.linalg.norm(tips[:, [0, 1, 2]] - tips[:, [1, 2, 4]], axis=-1)
        out[:] = np.concatenate((shape.reshape(SEQUENCE_FRAMES, -1), path, delta, gaps), axis=1).T
    return output


class FingerspellingSequence(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(FEATURE_CHANNELS, 96, kernel_size=3, padding=1),
            nn.BatchNorm1d(96), nn.GELU(),
            nn.Conv1d(96, 120, kernel_size=5, padding=2),
            nn.BatchNorm1d(120), nn.GELU(),
            nn.Conv1d(120, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.AdaptiveAvgPool1d(6),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 6, 128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, sequence):
        return self.classifier(self.encoder(sequence))
