"""Online Geometric & Temporal Skeleton Augmentations ported from KVSL.

Operates on canonical SSTCN tensors of shape [3, T, 75] (float32).
All transforms preserve missing joints (strictly 0.0) and never introduce NaN/Inf.
Horizontal mirroring is intentionally NOT applied because many VSL signs are
chirality-dependent (dominant hand and left->right motion semantics).
"""

import numpy as np

L_HAND_SLICE = slice(33, 54)
R_HAND_SLICE = slice(54, 75)


def _present_mask(seq_ctj: np.ndarray) -> np.ndarray:
    """Boolean mask [T, J] of non-zero joints."""
    return np.any(seq_ctj != 0.0, axis=0)


def augment_scale(
    seq_ctj: np.ndarray,
    rng: np.random.Generator,
    scale_range=(0.82, 1.22),
    is_v2: bool = True,
) -> np.ndarray:
    out = seq_ctj.copy()
    present = _present_mask(out)
    factor = float(rng.uniform(scale_range[0], scale_range[1]))
    if is_v2:
        out *= factor
    else:
        center = np.array([0.5, 0.5, 0.0], dtype=np.float32)[:, None, None]
        out = (out - center) * factor + center
    out[:, ~present] = 0.0
    return out


def augment_rotate_xy(
    seq_ctj: np.ndarray,
    rng: np.random.Generator,
    max_deg: float = 15.0,
    is_v2: bool = True,
) -> np.ndarray:
    out = seq_ctj.copy()
    present = _present_mask(out)
    angle = float(np.deg2rad(rng.uniform(-max_deg, max_deg)))
    c, s = float(np.cos(angle)), float(np.sin(angle))
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    if is_v2:
        out = np.einsum("ij,jtv->itv", rot, out)
    else:
        center = np.array([0.5, 0.5, 0.0], dtype=np.float32)[:, None, None]
        out = np.einsum("ij,jtv->itv", rot, out - center) + center
    out[:, ~present] = 0.0
    return out


def augment_translate(
    seq_ctj: np.ndarray,
    rng: np.random.Generator,
    max_shift: float = 0.08,
) -> np.ndarray:
    out = seq_ctj.copy()
    present = _present_mask(out)
    shift = rng.uniform(-max_shift, max_shift, size=(3, 1, 1)).astype(np.float32)
    shift[2] *= 0.5  # smaller z jitter
    out += shift
    out[:, ~present] = 0.0
    return out


def augment_gaussian_noise(
    seq_ctj: np.ndarray,
    rng: np.random.Generator,
    std: float = 0.012,
) -> np.ndarray:
    out = seq_ctj.copy()
    present = _present_mask(out)
    noise = rng.normal(0.0, std, size=out.shape).astype(np.float32)
    out += noise
    out[:, ~present] = 0.0
    return out


def augment_time_stretch(
    seq_ctj: np.ndarray,
    rng: np.random.Generator,
    rate_range=(0.75, 1.25),
) -> np.ndarray:
    """Simulate slower or faster signing speed via linear resampling + edge padding."""
    C, T, J = seq_ctj.shape
    rate = float(rng.uniform(rate_range[0], rate_range[1]))
    # Span inside [0, T-1] proportional to rate
    if rate >= 1.0:
        # Faster sign: active portion takes fewer frames, padded with start/end frames
        active_len = max(6, int(round(T / rate)))
        src_pos = np.linspace(0.0, T - 1, active_len)
        lo = src_pos.astype(int)
        hi = np.minimum(lo + 1, T - 1)
        w = (src_pos - lo).astype(np.float32)[None, :, None]
        compressed = seq_ctj[:, lo, :] * (1.0 - w) + seq_ctj[:, hi, :] * w
        pad_total = T - active_len
        pad_left = int(rng.integers(0, pad_total + 1)) if pad_total > 0 else 0
        pad_right = pad_total - pad_left
        return np.pad(compressed, ((0, 0), (pad_left, pad_right), (0, 0)), mode="edge").astype(np.float32)
    else:
        # Slower sign: take a contiguous crop of length rate*T and stretch to T
        crop_len = max(6, int(round(T * rate)))
        max_start = T - crop_len
        start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        src_pos = np.linspace(start, start + crop_len - 1, T)
        lo = src_pos.astype(int)
        hi = np.minimum(lo + 1, T - 1)
        w = (src_pos - lo).astype(np.float32)[None, :, None]
        return (seq_ctj[:, lo, :] * (1.0 - w) + seq_ctj[:, hi, :] * w).astype(np.float32)


def augment_temporal_shift(
    seq_ctj: np.ndarray,
    rng: np.random.Generator,
    max_shift: int = 3,
) -> np.ndarray:
    T = seq_ctj.shape[1]
    shift = int(rng.integers(-max_shift, max_shift + 1))
    idx = np.clip(np.arange(T) + shift, 0, T - 1)
    return seq_ctj[:, idx, :].copy()


def augment_joint_dropout(
    seq_ctj: np.ndarray,
    rng: np.random.Generator,
    drop_prob: float = 0.08,
) -> np.ndarray:
    out = seq_ctj.copy()
    mask = rng.random(out.shape[1:]) < drop_prob  # [T, J]
    out[:, mask] = 0.0
    return out


def augment_inter_hand_shift(
    seq_ctj: np.ndarray,
    rng: np.random.Generator,
    max_delta: float = 0.04,
) -> np.ndarray:
    """Slightly vary relative separation between left and right hands (KVSL inter-hand jitter)."""
    out = seq_ctj.copy()
    present = _present_mask(out)
    for sl in (L_HAND_SLICE, R_HAND_SLICE):
        hand_present = present[:, sl]
        if np.any(hand_present):
            delta = rng.uniform(-max_delta, max_delta, size=(3, 1, 1)).astype(np.float32)
            delta[2] *= 0.5
            out[:, :, sl] += delta
    out[:, ~present] = 0.0
    return out


def apply_kvsl_augmentations(
    seq_ctj: np.ndarray,
    rng: np.random.Generator,
    preprocessing_version: int = 2,
) -> np.ndarray:
    """Compose KVSL geometric and temporal augmentations safely on [3, T, 75]."""
    is_v2 = preprocessing_version >= 2
    out = np.asarray(seq_ctj, dtype=np.float32).copy()

    # Spatial transforms
    if rng.random() < 0.7:
        out = augment_scale(out, rng, scale_range=(0.85, 1.18) if is_v2 else (0.92, 1.08), is_v2=is_v2)
    if rng.random() < 0.7:
        out = augment_rotate_xy(out, rng, max_deg=14.0 if is_v2 else 8.0, is_v2=is_v2)
    if rng.random() < 0.5:
        out = augment_translate(out, rng, max_shift=0.06 if is_v2 else 0.03)
    if rng.random() < 0.4:
        out = augment_inter_hand_shift(out, rng, max_delta=0.035 if is_v2 else 0.015)

    # Temporal transforms
    if rng.random() < 0.6:
        out = augment_time_stretch(out, rng, rate_range=(0.80, 1.25))
    elif rng.random() < 0.6:
        out = augment_temporal_shift(out, rng, max_shift=3)

    # Noise & dropout
    if rng.random() < 0.6:
        out = augment_gaussian_noise(out, rng, std=0.012 if is_v2 else 0.004)
    if rng.random() < 0.35:
        out = augment_joint_dropout(out, rng, drop_prob=0.06)

    return np.ascontiguousarray(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float32)
