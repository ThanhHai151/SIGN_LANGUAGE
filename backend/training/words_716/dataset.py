"""
WordDataset: Skeletal Sequence Dataset with Controlled Domain Augmentation.
Dataset Contract: Input [3, 30, 75], Output Label in [0..715].
"""
from typing import List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset


class WordDataset(Dataset):
    def __init__(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        indices: Optional[List[int]] = None,
        augment: bool = False,
        seed: int = 42,
        scale_range: Tuple[float, float] = (0.95, 1.05),
        body_centric: bool = False,
    ):
        self.sequences = sequences
        self.labels = labels
        self.indices = list(indices) if indices is not None else list(range(len(labels)))
        self.augment = augment
        self.seed = seed
        self.epoch = 0
        self.scale_range = tuple(scale_range)
        self.body_centric = body_centric

    def __len__(self) -> int:
        return len(self.indices)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        row = self.indices[index]
        seq = np.array(self.sequences[row], copy=True)  # [3, 30, 75]
        target = int(self.labels[row])

        if self.body_centric:
            try:
                from backend.training.words_716.normalization import normalize_body_centric
            except ImportError:
                from normalization import normalize_body_centric
            seq, _ = normalize_body_centric(seq)

        if self.augment:
            # Deterministic RNG per (seed, epoch, row)
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, row]))
            
            # Find active joints across all time steps
            present = np.any(seq != 0, axis=0)  # [30, 75]
            
            # 1. Center coordinates around normalized image center [0.5, 0.5, 0.0]
            center = np.array([0.5, 0.5, 0.0], dtype=np.float32)[:, None, None]
            seq_centered = seq - center
            
            # 2. Slight planar 2D rotation (-0.10 to 0.10 rad, approx -5.7 to +5.7 degrees)
            angle = rng.uniform(-0.10, 0.10)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation = np.array(
                [[cos_a, -sin_a, 0.0],
                 [sin_a, cos_a, 0.0],
                 [0.0, 0.0, 1.0]],
                dtype=np.float32
            )
            # Apply rotation: [3, 3] x [3, 30, 75] -> [3, 30, 75]
            seq_rot = np.einsum("ij,jtv->itv", rotation, seq_centered)
            
            # 3. Spatial scale jitter (using configured scale_range)
            scale = rng.uniform(self.scale_range[0], self.scale_range[1])
            seq_scaled = seq_rot * scale + center
            
            # 4. Small landmark coordinate jitter (Gaussian sigma = 0.003)
            jitter = rng.normal(0.0, 0.003, seq.shape).astype(np.float32)
            seq_jittered = seq_scaled + jitter
            
            # 5. Strict preservation of missing/inactive joints (zero out non-present joints)
            seq_jittered[:, ~present] = 0.0
            
            # 6. Small temporal frame shift (-2 to +2 frames)
            shift = rng.integers(-2, 3)
            if shift != 0:
                frame_idx = np.clip(np.arange(30) + shift, 0, 29)
                seq_jittered = seq_jittered[:, frame_idx, :]
            
            seq = seq_jittered

        tensor = torch.from_numpy(np.ascontiguousarray(seq, dtype=np.float32))
        return tensor, target
