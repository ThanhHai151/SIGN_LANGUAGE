"""Compare a webcam recording with the selected QIPEDC video."""

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
TIME_SIGN_ENTRY = "q-may-gio"
TIME_SIGN_VIDEO = "W02196"
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


def _fill_short_gaps(clip):
    clip = clip.copy()
    for joints in (slice(11, 12), slice(12, 13)):
        points = clip[:, joints]
        valid = np.flatnonzero(np.any(points, axis=(1, 2)))
        for left, right in zip(valid, valid[1:]):
            gap = right - left - 1
            if 0 < gap <= 3:
                scale = np.arange(1, gap + 1, dtype=np.float32)[:, None, None] / (gap + 1)
                points[left + 1:right] = points[left] + scale * (points[right] - points[left])
    for joints, pose_wrist_idx in ((slice(33, 54), 15), (slice(54, 75), 16)):
        points = clip[:, joints]
        pose_wrist = clip[:, pose_wrist_idx]
        valid = np.flatnonzero(np.any(points, axis=(1, 2)))
        for left, right in zip(valid, valid[1:]):
            gap = right - left - 1
            if gap <= 0:
                continue
            scale = np.arange(1, gap + 1, dtype=np.float32)[:, None, None] / (gap + 1)
            # Bridge 1-2 frame fast-motion/rotation dropouts along the tracked pose wrist curve
            # when Holistic pose wrist is continuously present across the gap.
            if gap <= 2 and np.all(np.any(pose_wrist[left:right + 1] != 0, axis=1)):
                err_l = points[left, 0] - pose_wrist[left]
                err_r = points[right, 0] - pose_wrist[right]
                wrist_path = pose_wrist[left + 1:right] + err_l + scale[:, 0, :] * (err_r - err_l)
                shape_l = points[left] - points[left, :1]
                shape_r = points[right] - points[right, :1]
                points[left + 1:right] = wrist_path[:, None, :] + shape_l + scale * (shape_r - shape_l)
            elif gap <= 2:
                points[left + 1:right] = points[left] + scale * (points[right] - points[left])
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


def _time_sign_with_occlusion(body, hands, times, side, taps):
    """Verify wrist contact and the opening hand when the supporting hand drops out."""
    start = (33, 54)[side]
    hand = body[:, start:start + 21]
    fingers = hand[:, [8, 12, 16, 20]] - hand[:, :1]
    palm = np.maximum(np.linalg.norm(hand[:, 9] - hand[:, 0], axis=1), 1e-4)
    extension = np.linalg.norm(fingers, axis=2).mean(axis=1) / palm
    spread = np.linalg.norm(hand[:, 8] - hand[:, 20], axis=1) / palm
    # Check contact against at least one detected hand. The pose wrist can
    # corroborate nearby frames when the hand tracker loses the supporting hand.
    support_wrist = body[:, (54, 33)[side], :2]
    elbow = body[:, (14, 13)[side], :2]
    forearm = support_wrist - elbow
    contact = np.linalg.norm(hand[:, 8, :2] - support_wrist, axis=1)
    pose_wrist = body[:, (16, 15)[side], :2]
    pose_forearm = pose_wrist - elbow
    pose_contact = (hands[:, side] & np.any(pose_wrist != 0, axis=1) & np.any(elbow != 0, axis=1) &
                    (np.abs(pose_forearm[:, 0]) >= np.abs(pose_forearm[:, 1])) &
                    (np.linalg.norm(hand[:, 8, :2] - pose_wrist, axis=1) <= 0.50))
    for tap in taps:
        if contact[tap] > 0.30 or not elbow[tap].any() or abs(forearm[tap, 0]) < abs(forearm[tap, 1]):
            continue
        nearby = np.abs(times - times[tap]) <= 0.2
        if nearby[taps].sum() < 2 and (nearby & pose_contact).sum() < 2:
            continue
        elapsed = times - times[tap]
        tip_rise = hand[tap, 8, 1] - hand[:, 8, 1]
        opened = (hands[:, side] & (elapsed >= 0.18) & (elapsed <= 1.4) &
                  (tip_rise > 0.28) & (extension >= 1.55) &
                  (extension - extension[tap] >= 0.4) & (spread >= 1.2))
        for end in np.flatnonzero(opened[:-1] & opened[1:]) + 1:
            tracked = hands[tap:end + 1, side]
            if tracked.mean() >= 0.75 and np.diff(times[tap:end + 1][tracked]).max() <= 0.3:
                return int(tap), int(end)
    return None


def analyze_live_words(frames, aspect_ratio, enable_time_sign, timestamps=None):
    """Analyze a rolling window before the closed-set classifier guesses a word."""
    try:
        clip, first_frame = _visible_clip(frames)
    except ValueError:
        return {"motion_detected": False, "status": "insufficient_landmarks"}
    raw_hands = _tracked_hands(clip)
    clip = _fill_short_gaps(clip)
    clip[:, :, 1] /= aspect_ratio
    hands = _tracked_hands(clip)
    body = normalize_sequence_tjc_v2(clip)

    # Check hand motion relative to the shoulders, using [5, 95] percentiles so isolated
    # single-frame jitter is ignored while fast round-trip signs still pass the motion gate.
    motion = False
    for side, start in enumerate((33, 54)):
        tracked = body[hands[:, side], start:start + 21, :2]
        if len(tracked) >= 4:
            extent = np.percentile(tracked, 95, axis=0) - np.percentile(tracked, 5, axis=0)
            motion |= float(np.linalg.norm(extent, axis=1).mean()) >= 0.16
    if not motion:
        return {"motion_detected": False, "status": "no_motion"}
    if not enable_time_sign:
        return {"motion_detected": True, "status": "topic_excluded"}
    if not reference_path(TIME_SIGN_VIDEO).is_file():
        return {"motion_detected": True, "status": "reference_unavailable"}

    times = np.arange(len(clip), dtype=np.float64) / 15
    if timestamps is not None and len(timestamps) == len(frames):
        observed = np.asarray(timestamps[first_frame:first_frame + len(clip)], dtype=np.float64) / 1000
        if np.isfinite(observed).all() and (np.diff(observed) > 0).all():
            times = observed

    # The QIPEDC sign first points at the opposite wrist, then lifts the same hand.
    # Requiring their order rejects a held pose and either phase performed alone.
    first_tap = None
    phase_result = None
    for side, (tip, opposite_wrist) in enumerate(((41, 54), (62, 33))):
        index = body[:, tip, :2]
        other_tips = [tip + 4, tip + 8, tip + 12]
        fingers_visible = np.any(clip[:, other_tips, :] != 0, axis=2).all(axis=1)
        finger_height = np.median(body[:, other_tips, 1], axis=1)
        taps = np.flatnonzero(hands.all(axis=1) & (np.linalg.norm(index - body[:, opposite_wrist, :2], axis=1) < 0.50))
        for tap in taps:
            if first_tap is None or tap < first_tap:
                first_tap = tap
            after = slice(tap + 2, None)
            lifts = hands[after, side] & (index[tap, 1] - index[after, 1] > 0.28)
            # A frozen index tip can hide the lift even while the other fingers track correctly.
            if fingers_visible[tap] and len(lifts) >= 3:
                other_lifts = (hands[after, side] & fingers_visible[after] &
                               (finger_height[tap] - finger_height[after] > 0.28))
                lifts[:-2] |= other_lifts[:-2] & other_lifts[1:-1] & other_lifts[2:]
            if lifts.any():
                if phase_result is None:
                    match = compare_recording(TIME_SIGN_VIDEO, frames, aspect_ratio, either_hand=True)
                    phase_result = {"motion_detected": True,
                                    "status": "matched" if match["matched"] else match["reason"] or "low_similarity",
                                    "similarity": match["similarity"], "tap_frame": int(first_frame + tap),
                                    "lift_frame": int(first_frame + tap + 2 + np.flatnonzero(lifts)[0])}
                    if match["matched"]:
                        return phase_result
                recovered = _time_sign_with_occlusion(body, raw_hands, times, side, taps[raw_hands[taps].all(axis=1)])
                if recovered is not None:
                    contact, opened = recovered
                    return {"motion_detected": True, "status": "matched", "match_method": "phases",
                            "similarity": None, "tap_frame": first_frame + contact, "lift_frame": first_frame + opened}
                break
    if phase_result is not None:
        return phase_result
    if first_tap is not None:
        return {"motion_detected": True, "status": "lift_not_seen", "tap_frame": int(first_frame + first_tap)}
    return {"motion_detected": True, "status": "wrist_not_seen"}
