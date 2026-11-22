"""Shared Body-Centric Skeleton Normalization (Preprocessing v2) & Time Resampling.

Used identically by both training (prepare/train/finetune) and realtime inference
(VSLRecognizer) to guarantee zero train-serve skew.

Design lessons combined from KVSL, SignLang, and REPO:
- Keep canonical tensor layout [3, 30, 75] (33 pose + 21 left hand + 21 right hand)
  for full compatibility with Frontend MediaPipe Holistic & SSTCN_Attention.
- Zero lower-body pose joints (pose[23..32]) as in KVSL (UPPER_BODY_POSE_INDICES 0..22),
  removing webcam noise from unseen hips/legs.
- Center coordinates on shoulder midpoint (pose[11], pose[12]) and normalize by
  sequence-median 2D shoulder width for translation and scale invariance.
- Preserve missing landmarks strictly as 0.0.
"""

from typing import Sequence, Optional, Tuple, List
import numpy as np

LEFT_SHOULDER_IDX = 11
RIGHT_SHOULDER_IDX = 12
UPPER_BODY_POSE_END = 23  # keep pose 0..22, zero 23..32
POSE_END = 33
NUM_JOINTS = 75
TARGET_FRAMES = 30

PREPROCESSING_V2 = {
    "version": 2,
    "shape": [3, TARGET_FRAMES, NUM_JOINTS],
    "dtype": "float32",
    "joint_order": ["pose:0..32", "left_hand:0..20", "right_hand:0..20"],
    "coordinates": "body_centric_shoulder_normalized_xyz",
    "lower_body_zeroed": True,
    "missing_landmarks": "zero",
    "normalization": "shoulder_center_and_width",
    "temporal_sampling": "linear_resample_to_30_from_time_window",
}

# MediaPipe pairs in semantic left/right order. The nose stays at index zero.
MIRRORED_JOINTS = [
    0, 4, 5, 6, 1, 2, 3, 8, 7, 10, 9,
    12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21,
    24, 23, 26, 25, 28, 27, 30, 29, 32, 31,
    *range(54, 75), *range(33, 54),
]


def reflect_legacy_left_hand(frames: np.ndarray) -> np.ndarray:
    """Match the legacy word checkpoint's right-handed single-hand training data."""
    hands = np.any(frames[:, [33, 54], :] != 0, axis=2)
    if hands[:, 1].any() or hands[:, 0].mean() < 0.60:
        return frames
    reflected = frames[:, MIRRORED_JOINTS, :].copy()
    present = np.any(reflected != 0, axis=-1)
    reflected[..., 0] = np.where(present, 1.0 - reflected[..., 0], 0.0)
    return reflected


def get_preprocessing_version(preprocessing: Optional[dict]) -> int:
    """Return integer preprocessing contract version (defaults to 1 for legacy checkpoints)."""
    if not isinstance(preprocessing, dict):
        return 1
    try:
        return int(preprocessing.get("version", 1))
    except (TypeError, ValueError):
        return 1


def normalize_sequence_tjc_v2(seq_tjc: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Normalize a [T, 75, 3] skeleton sequence to body-centric shoulder-normalized coords.

    Properties:
    - Exact 3D translation invariance per frame (when shoulders/pose present).
    - Exact positive uniform scale invariance across the sequence.
    - Lower-body pose joints (23..32) are zeroed out.
    - Missing joints (all-zero before normalization) remain strictly 0.0.
    """
    arr = np.asarray(seq_tjc, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1:] != (NUM_JOINTS, 3) or arr.shape[0] == 0:
        raise ValueError(f"Expected [T, 75, 3] sequence, got {arr.shape}")

    out = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).copy()
    # 1. Zero lower body joints (pose 23..32)
    out[:, UPPER_BODY_POSE_END:POSE_END, :] = 0.0

    # 2. Identify valid (present) joints
    valid = np.any(out != 0.0, axis=-1)  # [T, 75]
    if not np.any(valid):
        return out

    T = out.shape[0]
    centers = np.zeros((T, 3), dtype=np.float32)
    shoulder_widths = []
    fallback_spans = []

    for t in range(T):
        l_ok = bool(valid[t, LEFT_SHOULDER_IDX])
        r_ok = bool(valid[t, RIGHT_SHOULDER_IDX])
        if l_ok and r_ok:
            p_l = out[t, LEFT_SHOULDER_IDX]
            p_r = out[t, RIGHT_SHOULDER_IDX]
            centers[t] = 0.5 * (p_l + p_r)
            w = float(np.linalg.norm(p_l[:2] - p_r[:2]))
            if w > eps:
                shoulder_widths.append(w)
        elif l_ok:
            centers[t] = out[t, LEFT_SHOULDER_IDX]
        elif r_ok:
            centers[t] = out[t, RIGHT_SHOULDER_IDX]
        else:
            pose_valid = valid[t, :UPPER_BODY_POSE_END]
            if np.any(pose_valid):
                centers[t] = out[t, :UPPER_BODY_POSE_END][pose_valid].mean(axis=0)
            elif np.any(valid[t]):
                centers[t] = out[t][valid[t]].mean(axis=0)

        if np.any(valid[t]):
            pts = out[t][valid[t]]
            span = float(np.max(np.ptp(pts[:, :2], axis=0)))
            if span > eps:
                fallback_spans.append(span)

    if shoulder_widths:
        scale = float(np.median(shoulder_widths))
    elif fallback_spans:
        scale = float(np.median(fallback_spans))
    else:
        scale = 1.0
    scale = max(scale, eps)

    # Apply translation + scale strictly to valid joints; missing joints stay 0.0
    out = (out - centers[:, None, :]) / scale
    out[~valid] = 0.0
    return np.ascontiguousarray(out, dtype=np.float32)


def normalize_sequence_ctj_v2(seq_ctj: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Normalize a [3, T, 75] canonical SSTCN sequence to v2 coordinates."""
    arr = np.asarray(seq_ctj, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[0] != 3 or arr.shape[2] != NUM_JOINTS:
        raise ValueError(f"Expected [3, T, 75] sequence, got {arr.shape}")
    tjc = arr.transpose(1, 2, 0)
    norm_tjc = normalize_sequence_tjc_v2(tjc, eps=eps)
    return np.ascontiguousarray(norm_tjc.transpose(2, 0, 1), dtype=np.float32)


def resample_sequence_linear(frames: np.ndarray, target_len: int = TARGET_FRAMES) -> np.ndarray:
    """Uniformly resample a [T, ...] sequence to target_len frames via linear interpolation."""
    arr = np.asarray(frames, dtype=np.float32)
    if len(arr) == 0:
        raise ValueError("Cannot resample an empty sequence")
    if len(arr) == target_len:
        return arr.copy()
    if len(arr) == 1:
        return np.repeat(arr, target_len, axis=0)
    positions = np.linspace(0.0, len(arr) - 1, target_len)
    lo = positions.astype(int)
    hi = np.minimum(lo + 1, len(arr) - 1)
    w = (positions - lo).astype(np.float32).reshape((-1,) + (1,) * (arr.ndim - 1))
    return (arr[lo] * (1.0 - w) + arr[hi] * w).astype(np.float32)


def resample_sequence_by_timestamps(
    frames: np.ndarray,
    timestamps: Sequence[float],
    target_len: int = TARGET_FRAMES,
) -> np.ndarray:
    """Resample frames according to their actual wall-clock/video timestamps (FPS-invariant)."""
    arr = np.asarray(frames, dtype=np.float32)
    ts = np.asarray(timestamps, dtype=np.float64)
    if len(arr) == 0 or len(arr) != len(ts):
        raise ValueError("frames and timestamps must be non-empty and have matching length")
    if len(arr) == 1 or ts[-1] <= ts[0]:
        return resample_sequence_linear(arr, target_len=target_len)

    target_ts = np.linspace(ts[0], ts[-1], target_len)
    idx_hi = np.searchsorted(ts, target_ts, side="left")
    idx_hi = np.clip(idx_hi, 1, len(ts) - 1)
    idx_lo = idx_hi - 1
    dt = np.maximum(ts[idx_hi] - ts[idx_lo], 1e-9)
    w = np.clip((target_ts - ts[idx_lo]) / dt, 0.0, 1.0).astype(np.float32)
    w = w.reshape((-1,) + (1,) * (arr.ndim - 1))
    return (arr[idx_lo] * (1.0 - w) + arr[idx_hi] * w).astype(np.float32)


def trim_trailing_hand_drop(
    frames: np.ndarray,
    timestamps: Optional[Sequence[float]] = None,
    min_len: int = 12,
    drop_threshold_y: float = 0.015,
) -> Tuple[np.ndarray, Optional[List[float]]]:
    """
    Cắt tỉa các frame buông/hạ tay (hand-drop exit frames) ở cuối chuỗi cử chỉ
    trước khi nội suy, nhằm bảo tồn hình thái chuyển động có nghĩa và tránh nhiễu
    làm giảm độ tin cậy của mô hình nhận diện.

    frames: mảng [T, 75, 3] hoặc [T, 225]
    timestamps: danh sách thời gian tương ứng (nếu có)
    min_len: độ dài tối thiểu được giữ lại (mặc định 12)
    drop_threshold_y: ngưỡng gia tốc hạ xuống của cổ tay (trục Y hướng xuống đất)
    """
    arr = np.asarray(frames, dtype=np.float32)
    T = len(arr)
    if T <= min_len:
        return arr, list(timestamps) if timestamps is not None else None

    # Đảm bảo hình dạng [T, 75, 3] để kiểm tra tọa độ
    orig_shape = arr.shape
    reshaped = arr.reshape(T, 75, 3) if arr.ndim == 2 and arr.shape[1] == 225 else arr
    if reshaped.ndim != 3 or reshaped.shape[1] < 75:
        return arr, list(timestamps) if timestamps is not None else None

    cut_idx = T
    max_trim = min(6, T - min_len)  # Cắt tỉa tối đa 6 frame buông tay

    for _ in range(max_trim):
        t = cut_idx - 1
        if cut_idx <= min_len:
            break

        prev_t = t - 1
        frame_curr = reshaped[t]
        frame_prev = reshaped[prev_t]

        # Kiểm tra năng lượng bàn tay: 33..53 (Trái), 54..74 (Phải)
        left_hand_energy = float(np.sum(np.abs(frame_curr[33:54])))
        right_hand_energy = float(np.sum(np.abs(frame_curr[54:75])))
        prev_left_energy = float(np.sum(np.abs(frame_prev[33:54])))
        prev_right_energy = float(np.sum(np.abs(frame_prev[54:75])))

        # Trường hợp 1: Frame đuôi mất sạch bàn tay (zeros) trong khi trước đó có cử chỉ tay
        if (left_hand_energy < 1e-4 and right_hand_energy < 1e-4):
            earlier_hand_energy = float(np.sum(np.abs(reshaped[:t, 33:75])))
            if earlier_hand_energy > 1e-3:
                cut_idx = t
                continue

        # Trường hợp 2: Bàn tay đang rơi tự do xuống dưới (wrist Y tăng nhanh)
        is_dropping = False

        # Tay phải (khớp 54 là cổ tay phải)
        if right_hand_energy > 1e-4 and prev_right_energy > 1e-4:
            dy_rwrist = float(frame_curr[54, 1] - frame_prev[54, 1])
            if dy_rwrist > drop_threshold_y:
                is_dropping = True

        # Tay trái (khớp 33 là cổ tay trái)
        if left_hand_energy > 1e-4 and prev_left_energy > 1e-4:
            dy_lwrist = float(frame_curr[33, 1] - frame_prev[33, 1])
            if dy_lwrist > drop_threshold_y:
                is_dropping = True

        # Cổ tay Pose (15 và 16) nếu tay hand không rõ
        if not is_dropping:
            dy_pose_lwrist = float(frame_curr[15, 1] - frame_prev[15, 1])
            dy_pose_rwrist = float(frame_curr[16, 1] - frame_prev[16, 1])
            if (dy_pose_lwrist > drop_threshold_y and frame_curr[15, 1] > 0.4) or \
               (dy_pose_rwrist > drop_threshold_y and frame_curr[16, 1] > 0.4):
                is_dropping = True

        if is_dropping:
            cut_idx = t
        else:
            # Nếu frame này không phải là hạ tay, dừng cắt tỉa ngay
            break

    trimmed_arr = arr[:cut_idx].copy()
    trimmed_ts = list(timestamps[:cut_idx]) if timestamps is not None else None
    return trimmed_arr, trimmed_ts

