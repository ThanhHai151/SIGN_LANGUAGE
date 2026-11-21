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


def _solve_2link_ik_xy(shoulder_xy, wrist_xy, upper_len, fore_len, old_elbow_xy):
    """Recompute anatomically valid 2D elbow position for a shifted wrist (ported from KVSL)."""
    vec = wrist_xy - shoulder_xy
    dist = float(np.linalg.norm(vec))
    max_reach = max(1e-4, (upper_len + fore_len) * 0.995)
    min_reach = abs(upper_len - fore_len) + 1e-4
    if dist < 1e-5:
        return old_elbow_xy
    clamped = float(np.clip(dist, min_reach, max_reach))
    ux, uy = vec[0] / dist, vec[1] / dist
    cos_a = float(np.clip((upper_len ** 2 + clamped ** 2 - fore_len ** 2) / (2.0 * upper_len * clamped), -1.0, 1.0))
    sin_a = float(np.sqrt(max(0.0, 1.0 - cos_a ** 2)))
    # Choose the bend side closest to the original elbow so the arm never flips inside-out
    cand1 = shoulder_xy + upper_len * np.array([ux * cos_a - uy * sin_a, uy * cos_a + ux * sin_a], dtype=np.float32)
    cand2 = shoulder_xy + upper_len * np.array([ux * cos_a + uy * sin_a, uy * cos_a - ux * sin_a], dtype=np.float32)
    return cand1 if np.linalg.norm(cand1 - old_elbow_xy) <= np.linalg.norm(cand2 - old_elbow_xy) else cand2


def augment_inter_hand_shift(
    seq_ctj: np.ndarray,
    rng: np.random.Generator,
    max_delta: float = 0.04,
) -> np.ndarray:
    """Shift left/right hands and recompute elbows via 2-link IK (KVSL inter_hand_distance)."""
    out = seq_ctj.copy()
    present = _present_mask(out)
    T = out.shape[1]
    arm_specs = (
        (L_HAND_SLICE, 11, 13, 15, 33),  # left shoulder, elbow, pose wrist, hand wrist
        (R_HAND_SLICE, 12, 14, 16, 54),  # right shoulder, elbow, pose wrist, hand wrist
    )
    for sl, sh_i, el_i, pw_i, hw_i in arm_specs:
        hand_present = present[:, sl]
        if not np.any(hand_present):
            continue
        delta = rng.uniform(-max_delta, max_delta, size=(3, 1, 1)).astype(np.float32)
        delta[2] *= 0.5
        out[:, :, sl] += delta
        if np.any(present[:, pw_i]):
            out[:, present[:, pw_i], pw_i] += delta[:, 0, 0:1]
        # Recompute elbow via 2-link IK wherever shoulder + elbow + wrist are present
        for t in range(T):
            if present[t, sh_i] and present[t, el_i] and (present[t, pw_i] or present[t, hw_i]):
                w_idx = pw_i if present[t, pw_i] else hw_i
                sh_xy = out[:2, t, sh_i]
                old_el_xy = out[:2, t, el_i]
                old_w_xy = out[:2, t, w_idx] - delta[:2, 0, 0]
                upper_len = float(np.linalg.norm(old_el_xy - sh_xy))
                fore_len = float(np.linalg.norm(old_w_xy - old_el_xy))
                if upper_len > 1e-4 and fore_len > 1e-4:
                    out[:2, t, el_i] = _solve_2link_ik_xy(sh_xy, out[:2, t, w_idx], upper_len, fore_len, old_el_xy)
    out[:, ~present] = 0.0
    return out


def augment_hand_occlusion(
    seq_ctj: np.ndarray,
    rng: np.random.Generator,
    min_frames: int = 2,
    max_frames: int = 6,
) -> np.ndarray:
    """Simulate brief single-hand self-occlusion (e.g. clasped hands) while keeping pose wrists."""
    out = seq_ctj.copy()
    present = _present_mask(out)
    l_ok = np.any(present[:, L_HAND_SLICE])
    r_ok = np.any(present[:, R_HAND_SLICE])
    if not (l_ok and r_ok):
        return out
    T = out.shape[1]
    span = int(rng.integers(min_frames, min(max_frames, max(min_frames, T // 3)) + 1))
    start = int(rng.integers(0, max(1, T - span + 1)))
    target_slice = L_HAND_SLICE if rng.random() < 0.5 else R_HAND_SLICE
    out[:, start:start + span, target_slice] = 0.0
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

    # Noise, dropout & brief single-hand self-occlusion
    if rng.random() < 0.6:
        out = augment_gaussian_noise(out, rng, std=0.012 if is_v2 else 0.004)
    if rng.random() < 0.35:
        out = augment_joint_dropout(out, rng, drop_prob=0.06)
    if rng.random() < 0.25:
        out = augment_hand_occlusion(out, rng, min_frames=2, max_frames=5)

    return np.ascontiguousarray(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float32)
