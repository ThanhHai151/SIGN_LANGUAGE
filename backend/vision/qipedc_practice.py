"""Compare a webcam recording with a QIPEDC reference video and gate live sign motion."""

from functools import lru_cache
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock

import numpy as np

from backend.models import video_cache
from backend.models.lexicon import motion_clip
from backend.vision.skeleton_norm import normalize_sequence_tjc_v2

FRAMES = 30
LIMIT = 0.28
_prepare_lock = Lock()


def reference_path(video_id):
    if not video_cache.ID_RE.fullmatch(video_id):
        raise ValueError("Mã video QIPEDC không hợp lệ")
    name = f"v3_{video_id}.npy"
    seed_dir = os.environ.get("VSL_QIPEDC_REFERENCE_DIR")
    if seed_dir:
        seed = Path(seed_dir) / name
        if seed.is_file():
            return seed
    return Path(video_cache.CACHE_DIR) / "practice" / name


def prepare_reference(video_id):
    path = reference_path(video_id)
    with _prepare_lock:
        if path.is_file():
            return
        from backend.training.ingest_videos import extract_video_to_75_joints

        video = video_cache.ensure_cached("videos", video_id)
        if not video:
            raise FileNotFoundError("Không tải được video mẫu từ QIPEDC.")
        active = motion_clip(video_id)
        try:
            frames = extract_video_to_75_joints(Path(video), target_len=None, active_window=active, webcam_profile=True)
            _visible_clip(frames)
        except ValueError:
            if not active:
                raise
            frames = extract_video_to_75_joints(Path(video), target_len=None, webcam_profile=True)
            _visible_clip(frames)
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=path.parent, suffix=".npy", delete=False) as tmp:
            temporary = Path(tmp.name)
            try:
                np.save(tmp, frames, allow_pickle=False)
                tmp.close()
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)


def _tracked_hands(frames):
    shoulders = np.all(np.any(frames[:, [11, 12], :] != 0, axis=2), axis=1)
    shoulders &= np.linalg.norm(frames[:, 11, :2] - frames[:, 12, :2], axis=1) > 1e-4
    return shoulders[:, None] & np.any(frames[:, [33, 54], :] != 0, axis=2)


def _visible_clip(frames):
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 3 or frames.shape[1:] != (75, 3) or len(frames) < 4 or not np.isfinite(frames).all():
        raise ValueError("Cần ít nhất 4 khung hình MediaPipe Holistic hợp lệ.")
    valid = np.flatnonzero(_tracked_hands(frames).any(axis=1))
    if len(valid) < 4:
        raise ValueError("Hãy giữ cả hai vai và bàn tay trong khung hình.")

    return frames[valid[0]:valid[-1] + 1], int(valid[0])


def _rotate_hand_2d(shape, angle_rad):
    if abs(angle_rad) < 1e-4:
        return shape
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    out = shape.copy()
    rx, ry = shape[:, 0], shape[:, 1]
    out[:, 0] = rx * c - ry * s
    out[:, 1] = rx * s + ry * c
    return out


def _fill_short_gaps(clip, *, allow_extended_pose_bridge=False):
    """Interpolate short landmark dropouts and optionally reconstruct pose-corroborated hand occlusion."""
    clip = clip.copy()
    for joints in (slice(11, 12), slice(12, 13)):
        points = clip[:, joints]
        valid = np.flatnonzero(np.any(points, axis=(1, 2)))
        for left, right in zip(valid, valid[1:]):
            gap = right - left - 1
            if 0 < gap <= 3:
                scale = np.arange(1, gap + 1, dtype=np.float32)[:, None, None] / (gap + 1)
                points[left + 1:right] = points[left] + scale * (points[right] - points[left])

    shoulders_ok = np.all(np.any(clip[:, [11, 12], :2] != 0, axis=2), axis=1)
    sw_arr = np.linalg.norm(clip[:, 11, :2] - clip[:, 12, :2], axis=1)
    sw_med = float(np.median(sw_arr[shoulders_ok])) if shoulders_ok.any() else 0.25
    sw_med = max(sw_med, 0.08)
    sh_y = np.where(shoulders_ok, clip[:, [11, 12], 1].mean(axis=1), 0.45)
    max_raised_y = sh_y + 0.90 * sw_med
    max_internal_pose_gap = 12 if allow_extended_pose_bridge else 2

    for joints, pose_wrist_idx, pose_index_idx in ((slice(33, 54), 15, 19), (slice(54, 75), 16, 20)):
        points = clip[:, joints]
        pose_wrist = clip[:, pose_wrist_idx]
        pose_index = clip[:, pose_index_idx]
        valid = np.flatnonzero(np.any(points, axis=(1, 2)))
        if len(valid) == 0:
            continue

        # 1. Internal gaps bounded by two real hand observations
        for left, right in zip(valid, valid[1:]):
            gap = right - left - 1
            if gap <= 0:
                continue
            scale = np.arange(1, gap + 1, dtype=np.float32)[:, None, None] / (gap + 1)
            pw_span = pose_wrist[left:right + 1]
            pw_valid = np.all(np.any(pw_span[:, :2] != 0, axis=1)) and np.all(
                pw_span[:, 1] <= max_raised_y[left:right + 1]
            )
            if gap <= max_internal_pose_gap and pw_valid:
                err_l = points[left, 0] - pose_wrist[left]
                err_r = points[right, 0] - pose_wrist[right]
                wrist_path = pose_wrist[left + 1:right] + err_l + scale[:, 0, :] * (err_r - err_l)
                shape_l = points[left] - points[left, :1]
                shape_r = points[right] - points[right, :1]
                points[left + 1:right] = wrist_path[:, None, :] + shape_l + scale * (shape_r - shape_l)
            elif gap <= 2:
                points[left + 1:right] = points[left] + scale * (points[right] - points[left])

        # 1.5 Repair physically impossible single-fingertip tracking spikes (>0.65 shoulder widths from MCP)
        valid_frames = np.flatnonzero(np.any(points, axis=(1, 2)))
        if len(valid_frames):
            max_finger_len = 0.65 * sw_med
            for mcp_i, pip_i, dip_i, tip_i in ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)):
                seg_vec = points[valid_frames, tip_i, :2] - points[valid_frames, mcp_i, :2]
                bad = np.linalg.norm(seg_vec, axis=1) > max_finger_len
                if np.any(bad):
                    bad_idx = valid_frames[bad]
                    points[bad_idx, tip_i] = points[bad_idx, dip_i] + (points[bad_idx, dip_i] - points[bad_idx, pip_i])

        if not allow_extended_pose_bridge or len(valid) < 2:
            continue

        # 2. Stationary passive-hand support (e.g., passive wrist/palm held still in signing space
        # while the active hand taps/slides over it and temporarily occludes its finger landmarks).
        first_v, last_v = int(valid[0]), int(valid[-1])
        pw_ok = np.any(pose_wrist[:, :2] != 0, axis=1) & (pose_wrist[:, 1] <= max_raised_y)
        if first_v > 0 and np.all(pw_ok[:first_v + 1]):
            lead_ptp = float(np.linalg.norm(np.ptp(pose_wrist[:first_v + 1, :2], axis=0)))
            anchor_err = float(np.linalg.norm(points[first_v, 0, :2] - pose_wrist[first_v, :2]))
            if lead_ptp <= 0.12 * sw_med and anchor_err <= 0.22 * sw_med:
                offset = points[first_v] - pose_wrist[first_v:first_v + 1]
                points[:first_v] = pose_wrist[:first_v, None, :] + offset[None, :, :]
        if last_v < len(clip) - 1 and np.all(pw_ok[last_v:]):
            tail_ptp = float(np.linalg.norm(np.ptp(pose_wrist[last_v:, :2], axis=0)))
            anchor_err = float(np.linalg.norm(points[last_v, 0, :2] - pose_wrist[last_v, :2]))
            if tail_ptp <= 0.12 * sw_med and anchor_err <= 0.22 * sw_med:
                base_shape = points[last_v] - points[last_v, :1]
                base_err = points[last_v, 0] - pose_wrist[last_v]
                pi_anchor_ok = np.any(pose_index[last_v, :2] != 0)
                base_ang = (
                    float(np.arctan2(pose_index[last_v, 1] - pose_wrist[last_v, 1], pose_index[last_v, 0] - pose_wrist[last_v, 0]))
                    if pi_anchor_ok else None
                )
                for t in range(last_v + 1, len(clip)):
                    rot_shape = base_shape
                    if base_ang is not None and np.any(pose_index[t, :2] != 0):
                        cur_ang = float(np.arctan2(pose_index[t, 1] - pose_wrist[t, 1], pose_index[t, 0] - pose_wrist[t, 0]))
                        delta = (cur_ang - base_ang + np.pi) % (2 * np.pi) - np.pi
                        rot_shape = _rotate_hand_2d(base_shape, float(np.clip(delta * 0.5, -0.40, 0.40)))
                    points[t] = (pose_wrist[t] + base_err)[None, :] + rot_shape
    return clip


def _features(clip):
    # Sample actual frames so missing hands stay zero instead of interpolating toward the origin.
    sample = clip[np.linspace(0, len(clip) - 1, FRAMES).round().astype(int)]
    present = _tracked_hands(sample)
    body = normalize_sequence_tjc_v2(sample)
    wrists = body[:, [33, 54], :2]
    hand = body[:, 33:, :].reshape(FRAMES, 2, 21, 3)
    shape = hand - hand[:, :, :1]
    palm = np.linalg.norm(shape[:, :, 9], axis=-1)
    shape = np.clip(shape / np.maximum(palm[:, :, None, None], 0.05), -3, 3)
    shape[..., 2] *= 0.4
    return wrists, shape, present


@lru_cache(maxsize=96)
def _reference_features(path):
    clip, _ = _visible_clip(np.load(path, allow_pickle=False))
    return _features(_fill_short_gaps(clip))


def _distance(sample, reference):
    sw, sh, sp = sample
    rw, rh, rp = reference
    common = sp[:, None] & rp[None, :]
    wrist = np.linalg.norm(sw[:, None] - rw[None, :], axis=-1)
    shape = np.linalg.norm(sh[:, None] - rh[None, :], axis=-1).mean(axis=-1)
    cost = ((0.65 * wrist + 0.35 * shape) * common).sum(axis=-1) / np.maximum(1, common.sum(axis=-1))
    cost += 0.28 * (sp[:, None] != rp[None, :]).sum(axis=-1)
    cost += 0.35 * ~common.any(axis=-1)
    # Limit the warp so a short fragment cannot stand in for the full sign.
    dp = np.full((FRAMES + 1, FRAMES + 1), np.inf)
    dp[0, 0] = 0
    for i in range(1, FRAMES + 1):
        for j in range(max(1, i - 8), min(FRAMES, i + 8) + 1):
            c = cost[i - 1, j - 1]
            dp[i, j] = min(dp[i - 1, j - 1] + 2 * c, dp[i - 1, j] + c, dp[i, j - 1] + c)
    return float(dp[-1, -1] / (2 * FRAMES))


def compare_recording(video_id, frames, aspect_ratio=1.0, mirror=False, either_hand=False):
    path = reference_path(video_id)
    if not path.is_file():
        raise FileNotFoundError("Cần chuẩn bị video mẫu trước khi ghi.")
    clip, _ = _visible_clip(frames)
    clip = _fill_short_gaps(clip)
    clip[:, :, 1] /= aspect_ratio
    hands = _tracked_hands(clip)
    reference = _reference_features(path)
    if hands.any(axis=1).mean() < 0.65:
        return {"matched": False, "similarity": None, "reason": "tracking_gaps"}
    if reference[2].all(axis=1).mean() >= 0.60 and hands.all(axis=1).mean() < 0.35:
        return {"matched": False, "similarity": None, "reason": "missing_both_hands"}
    references = [reference] if either_hand or not mirror else []
    if mirror or either_hand:
        wrists, shape, present = reference
        references.append((wrists[:, ::-1] * [-1, 1], shape[:, ::-1] * [-1, 1, 1], present[:, ::-1]))
    cuts = sorted({0, len(clip) // 10, len(clip) // 5})
    distance = min(
        _distance(sample, candidate) + penalty
        for sample, penalty in (
            (_features(clip[start:len(clip) - end]), 0.15 * (start + end) / len(clip))
            for start in cuts for end in cuts if len(clip) - start - end >= 4
        )
        for candidate in references
    )
    return {"matched": distance <= LIMIT, "similarity": round(max(0.0, 1.0 - distance), 3), "reason": None}


def analyze_live_words(frames, aspect_ratio, timestamps=None, trace=None, **_kwargs):
    """Sign-agnostic motion and landmark quality gate for rolling webcam windows."""
    del timestamps, _kwargs
    raw_len = len(frames) if frames is not None else 0
    try:
        clip, first_frame = _visible_clip(frames)
    except ValueError as exc:
        if trace is not None:
            trace.update({
                "raw_frames": raw_len,
                "visible_frames": 0,
                "first_visible_frame": None,
                "motion_threshold": 0.16,
                "motion_detected": False,
                "status": "insufficient_landmarks",
                "reason": str(exc),
            })
        return {"motion_detected": False, "status": "insufficient_landmarks"}
    raw_hands = _tracked_hands(clip)
    clip = _fill_short_gaps(clip)
    clip[:, :, 1] /= aspect_ratio
    hands = _tracked_hands(clip)
    body = normalize_sequence_tjc_v2(clip)

    # Check hand motion relative to the shoulders using [5, 95] percentiles so isolated
    # single-frame jitter is ignored while fast round-trip signs still pass the motion gate.
    motion = False
    side_extents = [0.0, 0.0]
    effective_threshold = 0.16
    for side, start in enumerate((33, 54)):
        tracked = body[hands[:, side], start:start + 21, :2]
        if len(tracked) >= 4:
            extent = np.percentile(tracked, 95, axis=0) - np.percentile(tracked, 5, axis=0)
            val = float(np.linalg.norm(extent, axis=1).mean())
            side_extents[side] = round(val, 4)
            wrist_y_med = float(np.median(tracked[:, 0, 1]))
            thr = 0.08 if wrist_y_med <= 0.55 and len(tracked) >= 6 else 0.16
            effective_threshold = min(effective_threshold, thr)
            motion |= val >= thr
    if trace is not None:
        shoulders_ok = np.all(np.any(clip[:, [11, 12], :] != 0, axis=2), axis=1)
        trace.update({
            "raw_frames": raw_len,
            "visible_frames": int(len(clip)),
            "first_visible_frame": int(first_frame),
            "shoulders_ratio": round(float(shoulders_ok.mean()), 3),
            "raw_left_frames": int(raw_hands[:, 0].sum()),
            "raw_right_frames": int(raw_hands[:, 1].sum()),
            "raw_both_frames": int(raw_hands.all(axis=1).sum()),
            "filled_left_frames": int(hands[:, 0].sum()),
            "filled_right_frames": int(hands[:, 1].sum()),
            "filled_both_frames": int(hands.all(axis=1).sum()),
            "left_extent": side_extents[0],
            "right_extent": side_extents[1],
            "max_extent": round(max(side_extents), 4),
            "motion_threshold": effective_threshold,
            "motion_detected": bool(motion),
        })
    if not motion:
        if trace is not None:
            trace["status"] = "no_motion"
        return {"motion_detected": False, "status": "no_motion"}
    if trace is not None:
        trace["status"] = "motion_detected"
    return {"motion_detected": True, "status": "motion_detected"}
