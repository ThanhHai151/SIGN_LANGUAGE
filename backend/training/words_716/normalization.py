"""
Body-Centric Normalization Module for WordRecognizer716 (TASK WORDS 3.2H.1).

Eliminates signer distance and camera framing domain shifts between studio (QIPEDC)
and arbitrary webcams while strictly preserving gesture kinematics and handshape geometry.

Invariants:
1. Input/Output Shape: [3, 30, 75] float32 (or [B, 3, 30, 75]).
2. Left/Right Hand Preservation: Joints 33..53 strictly Left Hand, 54..74 strictly Right Hand.
3. Isotropic Spatial Scaling: S_x == S_y, perfectly preserving finger length ratios and angles.
4. Missing Landmark Preservation: (0.0, 0.0, 0.0) remains strictly (0.0, 0.0, 0.0).
5. Sequence-Level Median Anchor: Eliminates high-frequency frame jitter and preserves natural
   whole-body communicative sway/leaning.
6. Local Hand Z Preservation: Hand Z represents wrist-relative finger curl depth and is
   scaled by S without translating origin.
"""
from typing import Dict, Optional, Tuple, Union
import numpy as np

# Canonical reference parameters empirically audited across 703 QIPEDC training samples
CANONICAL_SHOULDER_WIDTH: float = 0.1704
CANONICAL_MID_SHOULDER: Tuple[float, float] = (0.4978, 0.4985)
MIN_VALID_SHOULDER_FRAMES: int = 3
MIN_DEGENERATE_SHOULDER_WIDTH: float = 0.02


def compute_body_anchor(
    sequence: np.ndarray,
    canonical_sh_w: float = CANONICAL_SHOULDER_WIDTH,
    canonical_center: Tuple[float, float] = CANONICAL_MID_SHOULDER,
) -> Tuple[float, float, float, bool, int]:
    """
    Compute robust sequence-level median body center and shoulder width.

    Parameters:
        sequence: [3, 30, 75] or [T, 75, 3] numpy array
        canonical_sh_w: Default canonical shoulder width
        canonical_center: Default canonical center (X, Y)

    Returns:
        body_cx: float
        body_cy: float
        body_sh_w: float
        is_fallback: bool
        valid_shoulder_frames: int
    """
    if sequence.ndim != 3:
        raise ValueError(f"Expected 3D array [3, T, 75] or [T, 75, 3], got shape {sequence.shape}")

    # Canonicalize view to [3, T, 75]
    if sequence.shape[0] == 3 and sequence.shape[2] == 75:
        seq_3xtx75 = sequence
    elif sequence.shape[2] == 3 and sequence.shape[1] == 75:
        seq_3xtx75 = sequence.transpose(2, 0, 1)
    else:
        raise ValueError(f"Unsupported sequence shape {sequence.shape}")

    # Pose Shoulder landmarks: Left Shoulder = 11, Right Shoulder = 12
    p11 = seq_3xtx75[:2, :, 11]  # [2, T]
    p12 = seq_3xtx75[:2, :, 12]  # [2, T]

    # Shoulder frame is valid if both left and right shoulder X coordinates are non-zero
    valid_sh = (p11[0] != 0) & (p12[0] != 0)
    valid_count = int(np.sum(valid_sh))

    if valid_count >= MIN_VALID_SHOULDER_FRAMES:
        sh_widths = np.linalg.norm(p11[:, valid_sh] - p12[:, valid_sh], axis=0)
        mid_x = (p11[0, valid_sh] + p12[0, valid_sh]) / 2.0
        mid_y = (p11[1, valid_sh] + p12[1, valid_sh]) / 2.0

        body_sh_w = float(np.median(sh_widths))
        body_cx = float(np.median(mid_x))
        body_cy = float(np.median(mid_y))

        # Check for degenerate/collapsed shoulder width
        if body_sh_w < MIN_DEGENERATE_SHOULDER_WIDTH or not np.isfinite(body_sh_w):
            body_sh_w = canonical_sh_w
            is_fallback = True
        else:
            is_fallback = False
    else:
        # Fallback to Nose (joint 0) or canonical defaults
        nose = seq_3xtx75[:2, :, 0]
        valid_nose = nose[0] != 0
        if np.sum(valid_nose) >= MIN_VALID_SHOULDER_FRAMES:
            body_cx = float(np.median(nose[0, valid_nose]))
            body_cy = float(np.median(nose[1, valid_nose])) + (canonical_sh_w * 0.5)  # Offset nose to neck
        else:
            body_cx = canonical_center[0]
            body_cy = canonical_center[1]

        body_sh_w = canonical_sh_w
        is_fallback = True

    return body_cx, body_cy, body_sh_w, is_fallback, valid_count


def normalize_body_centric(
    sequence: np.ndarray,
    canonical_shoulder_width: float = CANONICAL_SHOULDER_WIDTH,
    canonical_center: Tuple[float, float] = CANONICAL_MID_SHOULDER,
) -> Tuple[np.ndarray, Dict[str, Union[float, int, bool]]]:
    """
    Apply sequence-level Body-Centric Normalization to a 75-landmark skeletal sequence.

    Transforms the signer's coordinates such that:
    1. Median mid-shoulder center is mapped to canonical_center (default: 0.4978, 0.4985).
    2. Median shoulder width is scaled to canonical_shoulder_width (default: 0.1704).
    3. Inactive/missing joints remain strictly 0.0.
    4. Handshape angles and ratios are strictly preserved via isotropic scaling.

    Parameters:
        sequence: np.ndarray with shape [3, 30, 75]
        canonical_shoulder_width: Target canonical shoulder width
        canonical_center: Target canonical (X, Y) center

    Returns:
        normalized_sequence: np.ndarray with identical shape [3, 30, 75] and dtype float32
        meta: Dictionary of extraction and transform statistics
    """
    is_transposed = False
    if sequence.shape[0] == 3 and sequence.shape[2] == 75:
        seq = np.array(sequence, copy=True, dtype=np.float32)
    elif sequence.shape[2] == 3 and sequence.shape[1] == 75:
        seq = np.array(sequence.transpose(2, 0, 1), copy=True, dtype=np.float32)
        is_transposed = True
    else:
        raise ValueError(f"Expected shape [3, 30, 75] or [30, 75, 3], received {sequence.shape}")

    # Track missing/inactive landmarks: joint is present if any of its 3 coordinates != 0
    present_mask = np.any(seq != 0.0, axis=0)  # [30, 75]

    body_cx, body_cy, body_sh_w, is_fallback, valid_sh_frames = compute_body_anchor(
        seq, canonical_sh_w=canonical_shoulder_width, canonical_center=canonical_center
    )

    scale_factor = canonical_shoulder_width / body_sh_w

    # Transform X and Y isotropically
    seq[0] = (seq[0] - body_cx) * scale_factor + canonical_center[0]
    seq[1] = (seq[1] - body_cy) * scale_factor + canonical_center[1]

    # Transform Z: Scale depth proportionally without translating origin (preserves hand local depth)
    seq[2] = seq[2] * scale_factor

    # Invariant: Restore exact zero coordinates for missing/inactive landmarks
    seq[:, ~present_mask] = 0.0

    meta = {
        "body_shoulder_width": round(body_sh_w, 4),
        "scale_factor": round(scale_factor, 4),
        "body_center_x": round(body_cx, 4),
        "body_center_y": round(body_cy, 4),
        "is_fallback": is_fallback,
        "valid_shoulder_frames": valid_sh_frames,
    }

    if is_transposed:
        seq = seq.transpose(1, 2, 0)

    return np.ascontiguousarray(seq, dtype=np.float32), meta


def normalize_batch(
    sequences: np.ndarray,
    canonical_shoulder_width: float = CANONICAL_SHOULDER_WIDTH,
    canonical_center: Tuple[float, float] = CANONICAL_MID_SHOULDER,
) -> np.ndarray:
    """
    Batch normalization helper for tensors of shape [N, 3, 30, 75].
    """
    if sequences.ndim != 4 or sequences.shape[1] != 3 or sequences.shape[3] != 75:
        raise ValueError(f"Expected batch tensor [N, 3, 30, 75], received {sequences.shape}")

    out = np.zeros_like(sequences, dtype=np.float32)
    for i in range(len(sequences)):
        norm_seq, _ = normalize_body_centric(
            sequences[i],
            canonical_shoulder_width=canonical_shoulder_width,
            canonical_center=canonical_center,
        )
        out[i] = norm_seq
    return out
