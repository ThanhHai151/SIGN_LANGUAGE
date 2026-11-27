"""Compare webcam motion with a library of prepared QIPEDC references using generic kinematics."""

from functools import lru_cache
import os
from pathlib import Path
import threading

import numpy as np

from backend.models import video_cache
from backend.vision.qipedc_practice import (
    _fill_short_gaps,
    _tracked_hands,
    _visible_clip,
    prepare_reference,
    reference_path,
)
from backend.vision.skeleton_norm import normalize_sequence_tjc_v2
from backend.vision.word_scope import verify_reference_video_id


DEFAULT_WORDS = (
    "q-may-gio",
    "q-an-du", "q-an-vua", "q-an-it", "q-com-2", "q-nem", "q-lau-mieng", "q-thom",
    "q-nan-ni", "q-cau-nguyen", "q-vai-lay",
    "q-xin-loi", "q-ky-nang", "q-kha-nang", "q-khong-co",
)
BUILTIN_WORDS = DEFAULT_WORDS
MAX_EXTRA_WORDS = 12
FRAMES = 24
COARSE_BINS = 6
FEATURE_DIM = 28
LEGACY_FEATURE_DIM = 14
STAGE1_TOP_K = 24
DTW_REFERENCE_BATCH_SIZE = 6
MAX_DISTANCE = 0.36
MAX_RARE_DISTANCE = 0.29
MIN_SEPARATION = 0.045
MAX_BILATERAL_DISTANCE = 0.52
MAX_RARE_BILATERAL_DISTANCE = 0.44
MIN_BILATERAL_SEPARATION = 0.075
MAX_HAND_SHAPE_DISTANCE = 0.36
MIN_HAND_SHAPE_SEPARATION = 0.045
MAX_UNI_CONTOUR_DISTANCE = 0.31
MIN_UNI_CONTOUR_SEPARATION = 0.040
MAX_BILATERAL_SHAPE_TRAJECTORY_DISTANCE = 0.60
MAX_UNILATERAL_SHAPE_TRAJECTORY_DISTANCE = 0.50
MAX_SINGLE_HAND_CORE_DISTANCE = 0.18
MIN_SINGLE_HAND_CORE_SEPARATION = 0.050
MAX_SINGLE_HAND_CORE_HINT_DISTANCE = 0.29
MAX_CORROBORATED_CORE_HINT_DISTANCE = 0.35
MAX_SIGNING_HEIGHT_SHIFT = 0.55
HAND_MISMATCH_PENALTY = 0.10
RAISED_WRIST_MAX_Y = 0.70

# 28-dim per-hand layout:
#  [0..1]   body-anchored wrist (wx, wy)
#  [2..3]   body-anchored fingertip centroid (tx, ty)
#  [4..7]   window-centered contour (dwx, dwy, dtx, dty)
#  [8..11]  signed temporal velocity (vwx, vwy, vtx, vty)
#  [12..21] normalized 2D 5-fingertip offsets from wrist (10 dims)
#  [22..27] mirror-invariant finger extension & thumb/spread scalars (6 dims: I, M, R, P, thumb, spread)
REFLECT_X = np.array(
    [*([-1.0, 1.0] * 11), 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    dtype=np.float32,
)
WRIST_WEIGHT = 0.48
FINGERTIP_WEIGHT = 0.72
CONTOUR_WEIGHT = 0.42
VELOCITY_WEIGHT = 1.15
TIP_SHAPE_WEIGHT = 0.18
FINGER_STATE_WEIGHT = 0.28

HAND_SHAPE_WEIGHTS = 0.78 * np.array(
    [
        0.55, 0.30,  # wrist (wx, wy)
        0.55, 0.30,  # tip centroid (tx, ty)
        0.75, 0.75, 0.85, 0.85,  # centered contour
        0.95, 0.95, 1.05, 1.05,  # signed velocity
        *([2.0] * 10),           # 5-tip 2D hand shape
        *([1.65] * 6),           # invariant finger extension/thumb/spread
    ],
    dtype=np.float32,
)

UNI_CONTOUR_WEIGHTS = 0.62 * np.array(
    [
        0.24, 0.48,  # attenuate lateral X arm-width offset while preserving head vs chest Y zone
        0.26, 0.52,
        1.45, 1.45, 1.55, 1.55,  # emphasize trajectory loop/arc contour
        1.35, 1.35, 1.45, 1.45,  # emphasize movement direction
        *([0.90] * 10),
        *([1.65] * 6),           # strongly discriminate 1/2/3/4/5 finger configurations
    ],
    dtype=np.float32,
)

UNI_CORE_WEIGHTS = np.repeat(np.array([0.40, 0.50, 0.20, 1.0], dtype=np.float32), [4, 4, 4, 16])

WARP_DIAGONALS = []
for _diagonal in range(2, 2 * FRAMES + 1):
    _i = np.arange(max(1, _diagonal - FRAMES), min(FRAMES, _diagonal - 1) + 1)
    _j = _diagonal - _i
    _within_band = np.abs(_i - _j) <= 5
    WARP_DIAGONALS.append((_i[_within_band], _j[_within_band]))

_DEFAULT_SEED_DIR = Path(__file__).resolve().parents[1] / "data" / "qipedc_reference"
_DEFAULT_CACHE_DIR = Path(video_cache.ROOT) / "data" / "media_cache"
_INDEX_NAME = "live_index.npz"


def lost_hand_during_interaction(raw, aspect_ratio):
    """Detect a two-hand interaction that continued with mostly single-hand tracking."""
    try:
        clip, _ = _visible_clip(raw)
    except ValueError:
        return False

    hands = _tracked_hands(clip)
    end = len(clip) - 1
    pair_offset = clip[:, 33, :2] - clip[:, 54, :2]
    pair_offset[:, 1] /= aspect_ratio
    shoulder_offset = clip[:, 11, :2] - clip[:, 12, :2]
    shoulder_offset[:, 1] /= aspect_ratio
    shoulder_width = np.linalg.norm(shoulder_offset, axis=1)
    close_pair = hands.all(axis=1) & (np.linalg.norm(pair_offset, axis=1) <= 0.85 * shoulder_width)

    shoulder_y = clip[:, [11, 12], 1].mean(axis=1)
    max_y = shoulder_y + 0.80 * shoulder_width * aspect_ratio
    pose_wrists = clip[:, [15, 16], :]
    pw_seen = np.any(pose_wrists != 0, axis=2)
    both_pw_seen = pw_seen.all(axis=1)
    pose_dropped = both_pw_seen & (pose_wrists[..., 1] >= max_y[:, None]).any(axis=1)
    hand_wrists = clip[:, [33, 54], :]
    any_raised = (
        (hands & (hand_wrists[..., 1] < max_y[:, None])).any(axis=1)
        | (both_pw_seen & (pose_wrists[..., 1] < max_y[:, None]).all(axis=1))
    )
    released = pose_dropped | ~any_raised
    breaks = np.flatnonzero(released[:-2] & released[1:-1] & released[2:])
    start = int(breaks[-1] + 3) if breaks.size else 0
    if start > end or close_pair[start:].sum() < 2:
        return False

    first_pair = start + int(np.flatnonzero(close_pair[start:])[0])
    tracked = hands[first_pair:]
    visible = tracked.any(axis=1)
    return bool(visible.sum() >= 8 and tracked[visible].all(axis=1).mean() < 0.60)


def _masked_velocity(seq, obs):
    """Compute smoothed central-difference velocity [U, H, D] respecting per-hand observation masks."""
    u, h, d = seq.shape
    vel = np.zeros((u, h, d), dtype=np.float32)
    if u < 2:
        return vel
    # Forward/backward adjacent differences where consecutive frames are observed
    adj_ok = obs[:-1] & obs[1:]
    step_diff = np.where(adj_ok[..., None], seq[1:] - seq[:-1], 0.0)
    vel[:-1] += step_diff
    vel[1:] += step_diff
    counts = np.zeros((u, h, 1), dtype=np.float32)
    counts[:-1] += adj_ok[..., None]
    counts[1:] += adj_ok[..., None]
    vel = np.where(counts > 0, vel / np.maximum(counts, 1.0), 0.0)
    if u >= 3:
        sm = vel.copy()
        sm[1:-1] = 0.25 * vel[:-2] + 0.50 * vel[1:-1] + 0.25 * vel[2:]
        vel = np.where(obs[..., None], sm, 0.0)
    return np.clip(vel, -0.30, 0.30)


def _compute_hand_features(hands_u, mouth_u, obs_u):
    """Build generic [U, H, 28] position + contour + velocity + 2D shape + invariant finger state."""
    u_len, n_hands, _, _ = hands_u.shape
    wrist_xy = hands_u[:, :, 0, :2]
    tips_xy = hands_u[:, :, [4, 8, 12, 16, 20], :2]
    wrist_rel = wrist_xy - mouth_u[:, None, :]
    # Use median across fingertips so a single noisy/frozen finger tip never corrupts hand trajectory
    tip_mean_rel = np.median(tips_xy, axis=2) - mouth_u[:, None, :]

    # Window-centered trajectory contour per hand
    w_contour = np.zeros_like(wrist_rel)
    t_contour = np.zeros_like(tip_mean_rel)
    for h_i in range(n_hands):
        seen_idx = obs_u[:, h_i]
        if seen_idx.any():
            w_med = np.median(wrist_rel[seen_idx, h_i], axis=0)
            t_med = np.median(tip_mean_rel[seen_idx, h_i], axis=0)
            w_contour[seen_idx, h_i] = wrist_rel[seen_idx, h_i] - w_med
            t_contour[seen_idx, h_i] = tip_mean_rel[seen_idx, h_i] - t_med

    vel_wt = _masked_velocity(np.concatenate((wrist_rel, tip_mean_rel), axis=-1), obs_u)

    raw_palm = np.linalg.norm(hands_u[:, :, 9] - hands_u[:, :, 0], axis=-1)
    med_tip_span = np.median(np.linalg.norm(tips_xy - wrist_xy[:, :, None, :], axis=-1), axis=2)
    palm_size = np.maximum(0.04, np.maximum(raw_palm, 0.55 * med_tip_span))
    shape_10 = np.clip(
        (tips_xy - wrist_xy[:, :, None, :]) / palm_size[:, :, None, None],
        -2.0,
        2.0,
    ).reshape(u_len, n_hands, 10)

    # Mirror- and rotation-invariant 5-finger extension + spread ratios
    finger_exts = []
    for ti, pi in ((8, 6), (12, 10), (16, 14), (20, 18)):
        d_tip = np.linalg.norm(hands_u[:, :, ti, :2] - wrist_xy, axis=-1) / palm_size
        d_pip = np.linalg.norm(hands_u[:, :, pi, :2] - wrist_xy, axis=-1) / palm_size
        finger_exts.append(np.clip(d_tip - d_pip, -0.60, 0.90))

    lat = hands_u[:, :, 5, :2] - hands_u[:, :, 17, :2]
    lat_unit = lat / np.maximum(np.linalg.norm(lat, axis=-1, keepdims=True), 1e-4)
    thumb_lat = np.sum((hands_u[:, :, 4, :2] - hands_u[:, :, 5, :2]) * lat_unit, axis=-1) / palm_size
    d_t9 = np.linalg.norm(hands_u[:, :, 4, :2] - hands_u[:, :, 9, :2], axis=-1) / palm_size
    thumb_open = np.clip(0.55 * thumb_lat + 0.45 * (d_t9 - 0.55), -0.60, 1.00)
    adj_gaps = np.stack(
        (
            np.linalg.norm(hands_u[:, :, 8, :2] - hands_u[:, :, 12, :2], axis=-1),
            np.linalg.norm(hands_u[:, :, 12, :2] - hands_u[:, :, 16, :2], axis=-1),
            np.linalg.norm(hands_u[:, :, 16, :2] - hands_u[:, :, 20, :2], axis=-1),
        ),
        axis=-1,
    )
    spread = 0.80 * np.clip(np.median(adj_gaps, axis=-1) / palm_size, 0.0, 1.50)
    finger_state_6 = np.stack((*finger_exts, thumb_open, spread), axis=-1)

    feats = np.concatenate(
        (
            WRIST_WEIGHT * wrist_rel,
            FINGERTIP_WEIGHT * tip_mean_rel,
            CONTOUR_WEIGHT * np.concatenate((w_contour, t_contour), axis=-1),
            VELOCITY_WEIGHT * vel_wt,
            TIP_SHAPE_WEIGHT * shape_10,
            FINGER_STATE_WEIGHT * finger_state_6,
        ),
        axis=-1,
    )
    return np.where(obs_u[..., None], feats, 0.0).astype(np.float32)


def _variants(raw, aspect_ratio, *, reference=False, trace=None):
    try:
        clip, _ = _visible_clip(raw)
    except ValueError as exc:
        if trace is not None:
            trace["variants_reject_reason"] = f"visible_clip_failed: {exc}"
        return None

    measured = _tracked_hands(clip)
    # Use pose-corroborated gap & passive-wrist bridging on live queries so self-occluded
    # two-hand signs remain bilaterally tracked without per-sign special cases.
    clip = _fill_short_gaps(clip, allow_extended_pose_bridge=not reference)
    clip[..., 1] /= aspect_ratio
    observed = _tracked_hands(clip)
    body = normalize_sequence_tjc_v2(clip)

    # Trim symmetric rest->sign->rest waist transitions (>0.70 at both ends); otherwise trim only deep hip drops (>1.12).
    w_min_y = np.where(observed, body[:, [33, 54], 1], np.inf).min(axis=1)
    both_ends_low = bool(len(w_min_y) >= 8 and w_min_y[0] > RAISED_WRIST_MAX_Y and w_min_y[-1] > RAISED_WRIST_MAX_Y)
    trim_thr = RAISED_WRIST_MAX_Y if both_ends_low else 1.12
    raised = observed & (body[:, [33, 54], 1] <= trim_thr)
    raised_idx = np.flatnonzero(raised.any(axis=1))
    if len(raised_idx) >= 7:
        sl = slice(int(raised_idx[0]), int(raised_idx[-1]) + 1)
        clip, observed, body, measured = clip[sl], observed[sl], body[sl], measured[sl]

    hands = body[:, 33:].reshape(len(body), 2, 21, 3)
    movement = np.zeros(2)
    wrist_y_med = np.full(2, np.inf)
    for side in (0, 1):
        points = hands[observed[:, side], side, :, :2]
        if len(points) >= 1:
            wrist_y_med[side] = float(np.median(points[:, 0, 1]))
        if len(points) >= 5:
            low, high = np.percentile(points, [5, 95], axis=0)
            movement[side] = np.linalg.norm(high - low, axis=-1).mean()

    min_move = np.array(
        [0.065 if wrist_y_med[side] <= 0.55 else 0.075 for side in (0, 1)],
        dtype=np.float32,
    )
    active = (movement >= min_move) & (movement >= 0.23 * max(movement.max(), 1e-6))
    # Ignore hip-hanging passive hand sway when the dominant hand is clearly signing above waist level
    if wrist_y_med.min() <= RAISED_WRIST_MAX_Y:
        active &= wrist_y_med <= 0.85
    obs_counts = observed.sum(axis=0)
    measured_pair = measured.all(axis=1)
    if not reference and active.all() and measured_pair.sum() < 3:
        # A short secondary track is not evidence of a two-hand sign. Keep two
        # sustained tracks when there may be real alternating/occluded hands.
        measured_counts = measured.sum(axis=0)
        if measured_counts.min() < max(5, int(np.ceil(0.20 * measured_counts.max()))):
            active[int(measured_counts.argmin())] = False
    if reference and not active.any():
        active = (obs_counts >= 5) & (wrist_y_med <= 0.82)
    forced_bilateral = False
    if not active.all() and (obs_counts >= 2).all():
        min_obs = int(obs_counts.min())
        sustained = (
            min_obs >= max(6, int(0.24 * len(clip)))
            and float(wrist_y_med.max()) <= RAISED_WRIST_MAX_Y
        )
        paired_mask = observed.all(axis=1) if reference else measured_pair
        close_clasp = False
        if min_obs >= 4 and paired_mask.sum() >= 3:
            wrist_gap = np.linalg.norm(hands[paired_mask, 0, 0, :2] - hands[paired_mask, 1, 0, :2], axis=-1)
            close_clasp = bool((wrist_gap <= 0.72).sum() >= 3)
        if (sustained or close_clasp) and (reference or measured_pair.sum() >= 2):
            forced_bilateral = True
            active[:] = True

    if trace is not None:
        trace.update({
            "left_movement": round(float(movement[0]), 4),
            "right_movement": round(float(movement[1]), 4),
            "movement_threshold": float(min_move.min()),
            "observed_counts": [int(obs_counts[0]), int(obs_counts[1])],
            "measured_pair_frames": int(measured_pair.sum()),
            "active_hands": "both" if active.all() else ("left" if active[0] else ("right" if active[1] else "none")),
            "bilateral_forced_by_2frame_rule": forced_bilateral,
        })
    observed = observed[:, active]
    tracked = np.flatnonzero(observed.any(axis=1))
    coverage = float(observed[tracked[0]:tracked[-1] + 1].any(axis=1).mean()) if len(tracked) else 0.0
    min_tracked = 4 if reference else 7
    min_cov = 0.35 if reference else 0.70
    if len(tracked) < min_tracked or coverage < min_cov:
        if trace is not None:
            trace.update({
                "tracked_frames": int(len(tracked)),
                "tracking_coverage": round(coverage, 3),
                "variants_reject_reason": (
                    "no_active_hand_below_0.09" if not active.any()
                    else ("too_few_tracked_frames_<7" if len(tracked) < min_tracked else "low_tracking_coverage_<0.70")
                ),
            })
        return None
    bilateral = active.all()

    mouth = body[:, [9, 10], :2].mean(axis=1)
    hands_act = hands[:, active]
    usable = slice(tracked[0], tracked[-1] + 1) if bilateral else tracked
    hands_u = hands_act[usable].copy()
    mouth_u = mouth[usable].copy()
    obs_u = observed[usable]

    if not bilateral and active[0]:
        hands_u[..., 0] *= -1.0
        mouth_u[:, 0] *= -1.0

    features = _compute_hand_features(hands_u, mouth_u, obs_u)
    observed = obs_u

    cut_div = 12 if reference else 6
    cuts = sorted({0, len(features) // cut_div})
    end_cuts = [0] if reference else cuts
    trims = [(start, end) for start in cuts for end in end_cuts if len(features) - start - end >= 4]
    penalty_list = [0.025 if start or end else 0.0 for start, end in trims]
    active_trim = None
    if not reference:
        active_trim = _active_motion_trim(features, observed)
        if active_trim is not None and active_trim not in trims and len(features) - active_trim[0] - active_trim[1] >= 4:
            trims.append(active_trim)
            penalty_list.append(0.015)
    if trace is not None:
        trace.update({
            "tracked_frames": int(len(tracked)),
            "tracking_coverage": round(coverage, 3),
            "usable_frames": int(len(features)),
            "reflected_left_hand": bool(not bilateral and active[0]),
            "active_trim": list(active_trim) if active_trim is not None else None,
            "query_variants": int(len(trims)),
        })
    indices = np.rint([
        np.linspace(start, len(features) - end - 1, FRAMES) for start, end in trims
    ]).astype(int)
    features, observed = features[indices], observed[indices]
    penalties = np.array(penalty_list, dtype=np.float32)
    if bilateral and reference:
        features = np.concatenate((features, features[:, :, ::-1] * REFLECT_X))
        observed = np.concatenate((observed, observed[:, :, ::-1]))
        penalties = np.tile(penalties, 2)
    return features, observed, penalties


def _active_motion_trim(features, observed, min_frames=8):
    """Isolate a fast or medium sign burst inside a longer idle-padded live window."""
    total = len(features)
    if total < min_frames + 4:
        return None
    pos = features[:, :, :4]
    both = observed[:-1] & observed[1:]
    diff = np.linalg.norm(pos[1:] - pos[:-1], axis=-1)
    speed = np.where(both, diff, 0.0).max(axis=1)
    peak = float(speed.max()) if len(speed) else 0.0
    if peak < 0.020:
        return None
    active = np.flatnonzero(speed >= max(0.016, 0.14 * peak))
    if len(active) < 2:
        return None
    start = max(0, int(active[0]) - 1)
    end = min(total, int(active[-1]) + 3)
    if end - start < min_frames:
        need = min_frames - (end - start)
        start = max(0, start - need // 2)
        end = min(total, start + min_frames)
        start = max(0, end - min_frames)
    end_cut = total - end
    if start + end_cut < max(3, total // 8):
        return None
    return (start, end_cut)


@lru_cache(maxsize=96)
def _reference_variants(path):
    return _variants(np.load(path, allow_pickle=False), 1, reference=True)


def prepare_live_reference(video_id):
    info = verify_reference_video_id(video_id)
    if not info["eligible"]:
        raise ValueError(f"Video không thuộc vùng miền cho phép ({info['reason']})")
    prepare_reference(video_id)
    return _reference_variants(reference_path(video_id)) is not None


def _variant_distances(recording, reference, *, phase_mode="full"):
    sample, sample_hands, cut = recording
    template, template_hands, ref_cut = reference
    n_sample, sample_frames, hands, feature_count = sample.shape
    n_template, template_frames, _, _ = template.shape
    source = sample.transpose(2, 0, 1, 3).reshape(hands, -1, feature_count)
    target = template.transpose(2, 0, 1, 3).reshape(hands, -1, feature_count)
    source_seen = sample_hands.transpose(2, 0, 1).reshape(hands, -1, 1)
    target_seen = template_hands.transpose(2, 0, 1).reshape(hands, 1, -1)
    together = source_seen & target_seen
    different = source_seen ^ target_seen
    source_squared = (source * source).sum(axis=2)
    target_squared = (target * target).sum(axis=2)
    squared_distances = source_squared[:, :, None] + target_squared[:, None, :] - 2 * (source @ target.transpose(0, 2, 1))
    delta = np.sqrt(np.maximum(squared_distances, 0))
    shared = together.sum(axis=0, dtype=np.float32)
    cost = (np.minimum(delta, 3) * together).sum(axis=0) + HAND_MISMATCH_PENALTY * different.sum(axis=0, dtype=np.float32)
    cost /= np.maximum(1, shared)
    cost += 0.7 * (shared == 0)
    cost = cost.reshape(n_sample, sample_frames, n_template, template_frames).transpose(0, 2, 1, 3)
    dp = np.full((FRAMES + 1, FRAMES + 1, len(sample), len(template)), np.inf, dtype=np.float32)
    dp[0, 0] = 0
    for i, j in WARP_DIAGONALS:
        dt = cost[:, :, i - 1, j - 1].transpose(2, 0, 1)
        dp[i, j] = np.minimum(dp[i - 1, j - 1] + dt, np.minimum(dp[i - 1, j], dp[i, j - 1])) + dt
    phase_penalty = 0.0
    if phase_mode == "full":
        per_hand = np.where(together, np.minimum(delta, 3), 0.0).reshape(
            hands, n_sample, sample_frames, n_template, template_frames
        )
        max_hand_cost = per_hand.max(axis=0).transpose(0, 2, 1, 3)
        start_mismatch = max_hand_cost[:, :, :2, :2].min(axis=(2, 3))
        end_mismatch = max_hand_cost[:, :, -2:, -2:].min(axis=(2, 3))
        phase_penalty = 0.85 * (np.maximum(0.0, start_mismatch - 0.28) + np.maximum(0.0, end_mismatch - 0.28))
    elif phase_mode == "finger_end":
        s_end = sample[:, -2:, :, 12:]
        t_end = template[:, -3:, :, 12:]
        s_ok = sample_hands[:, -2:, :]
        t_ok = template_hands[:, -3:, :]
        # Compare terminal intrinsic finger configuration (dims 12..28), invariant to signing height
        f_diff = np.linalg.norm(s_end[:, :, None, None, :, :] - t_end[None, None, :, :, :, :], axis=-1)
        both_ok = s_ok[:, :, None, None, :] & t_ok[None, None, :, :, :]
        masked_f = np.where(both_ok, f_diff, 0.0).max(axis=-1)
        end_finger_mismatch = masked_f.min(axis=(1, 3))
        phase_penalty = 0.70 * np.maximum(0.0, end_finger_mismatch - 0.55)
    return dp[-1, -1] / (2 * FRAMES) + phase_penalty + cut[:, None] + ref_cut


def _score_templates(recording, references, *, phase_mode="full"):
    hands = recording[0].shape[2]
    usable = [(entry, ref) for entry, ref in references.items() if ref[0].shape[2] == hands]
    scores = {}
    for offset in range(0, len(usable), DTW_REFERENCE_BATCH_SIZE):
        batch = usable[offset:offset + DTW_REFERENCE_BATCH_SIZE]
        ids, templates = zip(*batch)
        combined = tuple(np.concatenate(parts, axis=0) for parts in zip(*templates))
        first_variant = np.cumsum([0, *(len(ref[0]) for ref in templates[:-1])])
        distances = np.minimum.reduceat(
            _variant_distances(recording, combined, phase_mode=phase_mode), first_variant, axis=1
        )
        for entry_id, dist in zip(ids, distances.T):
            if entry_id not in scores or dist.min() < scores[entry_id].min():
                scores[entry_id] = dist
    return scores


def _relative_hand_features(features, observed):
    """Emphasize how the wrists, trajectories, and finger shapes relate between the two hands."""
    seen = observed[..., None]
    center = (features * seen).sum(axis=2, keepdims=True) / np.maximum(1, seen.sum(axis=2, keepdims=True))
    rel = features.copy()
    # Subtract inter-hand center only on body-anchored coordinates [0..3]; keep contour/velocity/shape intact!
    rel[..., :4] = features[..., :4] - 0.85 * center[..., :4].astype(features.dtype)
    return rel * HAND_SHAPE_WEIGHTS


def _unilateral_contour_features(features, observed):
    """Attenuate static lateral arm-width offset while emphasizing loop/arc contour, velocity, and 5-finger state."""
    return np.where(observed[..., None], features * UNI_CONTOUR_WEIGHTS, 0.0)


def _unilateral_core_features(features, observed):
    # Finger configuration is generally more stable than the speed of the
    # wrist or its exact distance from the camera between different signers.
    return np.where(observed[..., None], features * UNI_CORE_WEIGHTS, 0.0)


def _duplicate_hand_fraction(features, observed):
    gap = np.linalg.norm(features[:, :, 0, :4] - features[:, :, 1, :4], axis=-1)
    return float((observed.all(axis=2) & (gap < 0.08)).mean())


def _coarse_pool(features, observed):
    v, t, h, d = features.shape
    step = t // COARSE_BINS
    f = features.reshape(v, COARSE_BINS, step, h, d)
    o = observed.reshape(v, COARSE_BINS, step, h, 1).astype(np.float32)
    return (f * o).sum(axis=2) / np.maximum(1.0, o.sum(axis=2))


def index_path():
    seed_dir = os.environ.get("VSL_QIPEDC_REFERENCE_DIR")
    if seed_dir:
        candidate = Path(seed_dir) / _INDEX_NAME
        if candidate.is_file():
            return candidate
    cache_candidate = Path(video_cache.CACHE_DIR) / "practice" / _INDEX_NAME
    if cache_candidate.is_file():
        return cache_candidate
    if Path(video_cache.CACHE_DIR).resolve() == _DEFAULT_CACHE_DIR.resolve():
        default_candidate = _DEFAULT_SEED_DIR / _INDEX_NAME
        if default_candidate.is_file():
            return default_candidate
    return None


def _transform_coarse_chunked(features, observed, transform_fn, chunk_size=1024):
    total, _, hands, dims = features.shape
    out = np.empty((total, COARSE_BINS, hands, dims), dtype=np.float32)
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        f_chunk = features[start:end].astype(np.float32, copy=False)
        o_chunk = observed[start:end]
        out[start:end] = _coarse_pool(transform_fn(f_chunk, o_chunk), o_chunk)
    return out


def _slice_sub_bucket(bucket, keep_set):
    keep_entries = [i for i, eid in enumerate(bucket["ids"]) if eid in keep_set]
    if not keep_entries or len(keep_entries) == len(bucket["ids"]):
        return None
    sub_ids = [bucket["ids"][i] for i in keep_entries]
    var_indices = []
    sub_v2e = []
    sub_slices = []
    offset = 0
    for new_idx, old_idx in enumerate(keep_entries):
        st, en = bucket["slices"][old_idx]
        count = en - st
        var_indices.append(np.arange(st, en, dtype=np.int32))
        sub_v2e.append(np.full(count, new_idx, dtype=np.int32))
        sub_slices.append((offset, offset + count))
        offset += count
    v_idx = np.concatenate(var_indices)
    sub = {
        "ids": sub_ids,
        "features": bucket["features"][v_idx],
        "observed": bucket["observed"][v_idx],
        "penalties": bucket["penalties"][v_idx],
        "var_to_entry": np.concatenate(sub_v2e),
        "coarse": bucket["coarse"][v_idx],
        "coarse_squared": bucket["coarse_squared"][v_idx],
        "slices": sub_slices,
    }
    for key in ("relative_coarse", "contour_coarse", "core_coarse"):
        if key in bucket:
            sub[key] = bucket[key][v_idx]
            sub[key + "_squared"] = bucket[key + "_squared"][v_idx]
    return sub


@lru_cache(maxsize=1)
def _default_priority_ids():
    from backend.models.lexicon import load_lexicon

    lex = load_lexicon()
    return frozenset(
        eid for eid, entry in lex["by_id"].items() if entry.get("common")
    ) | frozenset(BUILTIN_WORDS)


def _project_idle_waist_bilateral_to_uni(uni_bucket, bi_bucket, *, max_idle_range=0.18):
    """Generic unilateral projection for studio 2-hand entries whose passive hand hangs idle near the waist."""
    existing = set(uni_bucket["ids"])
    bi_feats = bi_bucket["features"]
    bi_obs = bi_bucket["observed"]
    bi_pens = bi_bucket["penalties"]
    bi_coarse = bi_bucket["coarse"]
    reflect_f16 = REFLECT_X.astype(bi_feats.dtype)

    extra_ids = []
    f_parts, o_parts, p_parts, c_parts, v2e_parts = [], [], [], [], []
    next_entry_idx = len(uni_bucket["ids"])
    next_var_offset = len(uni_bucket["features"])

    for ei, eid in enumerate(bi_bucket["ids"]):
        if eid in existing:
            continue
        st, en = bi_bucket["slices"][ei]
        half = (en - st) // 2
        if half < 1:
            continue
        f0 = bi_feats[st]
        o0 = bi_obs[st]
        obs_n = o0.sum(axis=0)
        wy_med = [9.9, 9.9]
        w_rng = [0.0, 0.0]
        for s in (0, 1):
            if obs_n[s] >= 4:
                pts = f0[o0[:, s], s, :2].astype(np.float32) / WRIST_WEIGHT
                wy_med[s] = float(np.median(pts[:, 1]))
                w_rng[s] = float(np.linalg.norm(np.ptp(pts, axis=0)))
        act = 0 if wy_med[0] < wy_med[1] else 1
        idle = 1 - act
        if not (obs_n[act] >= 12 and wy_med[act] <= 1.30):
            continue
        if not (obs_n[idle] < 4 or (wy_med[idle] >= 1.42 and w_rng[idle] <= max_idle_range)):
            continue

        f_sub = bi_feats[st:st + half, :, act:act + 1, :]
        c_sub = bi_coarse[st:st + half, :, act:act + 1, :]
        if act == 0:
            f_sub = f_sub * reflect_f16
            c_sub = c_sub * REFLECT_X
        extra_ids.append(eid)
        f_parts.append(f_sub)
        o_parts.append(bi_obs[st:st + half, :, act:act + 1])
        p_parts.append(bi_pens[st:st + half])
        c_parts.append(c_sub)
        v2e_parts.append(np.full(half, next_entry_idx, dtype=np.int32))
        uni_bucket["slices"].append((next_var_offset, next_var_offset + half))
        next_entry_idx += 1
        next_var_offset += half

    if extra_ids:
        uni_bucket["ids"].extend(extra_ids)
        uni_bucket["features"] = np.concatenate([uni_bucket["features"], *f_parts], axis=0)
        uni_bucket["observed"] = np.concatenate([uni_bucket["observed"], *o_parts], axis=0)
        uni_bucket["penalties"] = np.concatenate([uni_bucket["penalties"], *p_parts], axis=0)
        uni_bucket["coarse"] = np.concatenate([uni_bucket["coarse"], *c_parts], axis=0)
        uni_bucket["var_to_entry"] = np.concatenate([uni_bucket["var_to_entry"], *v2e_parts], axis=0)


def _expand_legacy_features(features):
    # The published index stores wrist position, mean fingertip position, and
    # five palm-normalized fingertip offsets. Recover the median centroid and
    # estimate palm scale from the two means. Joint-dependent channels are absent.
    expanded = np.zeros((*features.shape[:-1], FEATURE_DIM), dtype=np.float16)
    for start in range(0, len(features), 1024):
        old = features[start:start + 1024].astype(np.float32)
        wrist = old[..., :2] / 0.60
        mean_tip = old[..., 2:4] / 0.95
        tips = old[..., 4:].reshape(*old.shape[:-1], 5, 2) / 0.22
        mean_offset = tips.mean(axis=-2)
        palm = np.clip(
            ((mean_tip - wrist) * mean_offset).sum(axis=-1)
            / np.maximum(1e-5, (mean_offset * mean_offset).sum(axis=-1)),
            0.04, 1.5,
        )
        palm_correction = np.maximum(1.0, 0.55 * np.median(np.linalg.norm(tips, axis=-1), axis=-1))
        out = expanded[start:start + len(old)]
        out[..., :2] = WRIST_WEIGHT * wrist
        out[..., 2:4] = FINGERTIP_WEIGHT * (wrist + np.median(tips, axis=-2) * palm[..., None])
        out[..., 12:22] = TIP_SHAPE_WEIGHT * np.clip(
            tips.reshape(*old.shape[:-1], 10) / palm_correction[..., None], -2.0, 2.0,
        )
    return expanded


def _index_feature_view(reference, index):
    if index is None or not index["legacy"]:
        return reference
    features, observed, penalties = reference
    comparable = features.copy()
    comparable[..., 4:12] = 0
    comparable[..., 22:] = 0
    return comparable, observed, penalties


@lru_cache(maxsize=2)
def _load_index_cached(path_str, mtime_ns):
    del mtime_ns
    priority_set = _default_priority_ids()
    with np.load(path_str, allow_pickle=False) as data:
        buckets = {}
        all_ids = []
        feature_dim = data["uni_features"].shape[-1]
        legacy = feature_dim == LEGACY_FEATURE_DIM
        for hands, prefix in ((1, "uni"), (2, "bi")):
            ids = [str(x) for x in data[f"{prefix}_ids"].tolist()]
            feats = data[f"{prefix}_features"]
            if feats.shape[-1] != feature_dim or feature_dim not in (LEGACY_FEATURE_DIM, FEATURE_DIM):
                raise ValueError(f"Incompatible live_index feature layout: {feats.shape[-1]}")
            all_ids.extend(ids)
            obs = data[f"{prefix}_observed"].astype(bool)
            if legacy:
                feats = _expand_legacy_features(feats)
            pens = data[f"{prefix}_penalties"].astype(np.float32)
            v2e = data[f"{prefix}_var_to_entry"].astype(np.int32)
            coarse = _coarse_pool(feats, obs) if legacy else data[f"{prefix}_coarse"].astype(np.float32)
            slices = []
            if len(ids):
                boundaries = np.flatnonzero(np.diff(v2e)) + 1
                starts = np.r_[0, boundaries]
                ends = np.r_[boundaries, len(v2e)]
                slices = list(zip(starts.tolist(), ends.tolist()))
            buckets[hands] = {
                "ids": ids, "features": feats, "observed": obs,
                "penalties": pens, "var_to_entry": v2e, "coarse": coarse, "slices": slices,
            }
        # The previous index includes more of the lead-in and has a wider
        # calibrated range for the passive hand than the newer clipped index.
        _project_idle_waist_bilateral_to_uni(
            buckets[1], buckets[2], max_idle_range=0.16 / 0.60 if legacy else 0.18,
        )
        buckets[1]["contour_coarse"] = _transform_coarse_chunked(
            buckets[1]["features"], buckets[1]["observed"], _unilateral_contour_features
        )
        buckets[1]["core_coarse"] = _transform_coarse_chunked(
            buckets[1]["features"], buckets[1]["observed"], _unilateral_core_features
        )
        buckets[2]["relative_coarse"] = _transform_coarse_chunked(
            buckets[2]["features"], buckets[2]["observed"], _relative_hand_features
        )
        for hands in (1, 2):
            bucket = buckets[hands]
            for key in ("coarse", "relative_coarse", "contour_coarse", "core_coarse"):
                if key in bucket:
                    bucket[key + "_squared"] = np.einsum("vthd,vthd->vth", bucket[key], bucket[key])
            bucket["priority_sub"] = _slice_sub_bucket(bucket, priority_set)
    return {"buckets": buckets, "all_ids": tuple(all_ids), "id_set": frozenset(all_ids), "legacy": legacy}


_INDEX_LOAD_LOCK = threading.Lock()


def _active_index():
    path = index_path()
    if path is None:
        return None
    try:
        with _INDEX_LOAD_LOCK:
            return _load_index_cached(str(path), path.stat().st_mtime_ns)
    except (OSError, ValueError, KeyError):
        return None


def indexed_entry_ids():
    idx = _active_index()
    return idx["all_ids"] if idx is not None else ()


def _stage1_candidates(recording, bucket, top_k=STAGE1_TOP_K, *, coarse_key="coarse", allowed_ids=None):
    n_entries = len(bucket["ids"])
    if n_entries == 0:
        return []
    if allowed_ids is not None:
        allowed_idx = [i for i, eid in enumerate(bucket["ids"]) if eid in allowed_ids]
        if not allowed_idx:
            return []
        if len(allowed_idx) <= top_k:
            return allowed_idx
    else:
        allowed_idx = None
        if n_entries <= top_k:
            return list(range(n_entries))

    sample_f, sample_o, sample_cut = recording
    sample_coarse = _coarse_pool(sample_f, sample_o)
    slice_allowed = allowed_idx is not None and len(allowed_idx) < (n_entries // 2)
    if slice_allowed:
        slices = bucket["slices"]
        var_idx_chunks = []
        sub_v2e_chunks = []
        for local_pos, entry_i in enumerate(allowed_idx):
            start, end = slices[entry_i]
            var_idx_chunks.append(np.arange(start, end, dtype=np.int32))
            sub_v2e_chunks.append(np.full(end - start, local_pos, dtype=np.int32))
        var_indices = np.concatenate(var_idx_chunks)
        if not len(var_indices):
            return []
        sub_v2e = np.concatenate(sub_v2e_chunks)
        tmpl_coarse = bucket[coarse_key][var_indices]
        tmpl_squared = bucket[coarse_key + "_squared"][var_indices]
        tmpl_pen = bucket["penalties"][var_indices]
        num_pool = len(allowed_idx)
    else:
        sub_v2e = bucket["var_to_entry"]
        tmpl_coarse = bucket[coarse_key]
        tmpl_squared = bucket[coarse_key + "_squared"]
        tmpl_pen = bucket["penalties"]
        num_pool = n_entries

    best_var = np.full(len(tmpl_coarse), np.inf, dtype=np.float32)
    for sv, cut in zip(sample_coarse, sample_cut):
        dot = np.einsum("vthd,thd->vth", tmpl_coarse, sv)
        squared = tmpl_squared + (sv * sv).sum(axis=-1) - 2 * dot
        diff = np.sqrt(np.maximum(squared, 0)).mean(axis=(1, 2)) + tmpl_pen + float(cut)
        np.minimum(best_var, diff, out=best_var)
    entry_best = np.full(num_pool, np.inf, dtype=np.float32)
    np.minimum.at(entry_best, sub_v2e, best_var)
    if slice_allowed or allowed_idx is not None:
        arr = entry_best if slice_allowed else entry_best[allowed_idx]
        chosen = np.argpartition(arr, top_k)[:top_k]
        return [allowed_idx[int(k)] for k in chosen]
    return np.argpartition(entry_best, top_k)[:top_k].tolist()


def _indexed_references(recording, bucket, *, coarse_key="coarse", allowed_ids=None):
    for entry_idx in _stage1_candidates(recording, bucket, coarse_key=coarse_key, allowed_ids=allowed_ids):
        start, end = bucket["slices"][entry_idx]
        yield bucket["ids"][entry_idx], (
            bucket["features"][start:end].astype(np.float32, copy=False),
            bucket["observed"][start:end],
            bucket["penalties"][start:end],
        )


def _match_result(score, entry_id, method="trajectory"):
    return {
        "motion_detected": True,
        "status": "word_matched",
        "entry_id": entry_id,
        "match_method": method,
        "similarity": round(max(0.0, 1.0 - score), 3),
    }


def _bilateral_tip_gap(features, observed):
    """Height-invariant median distance between left and right fingertip centroids on paired frames."""
    f0, o0 = features[0], observed[0]
    both = o0.all(axis=1)
    if not both.any():
        return None, 0.0
    tgap = float(np.median(np.linalg.norm(f0[both, 0, 2:4] - f0[both, 1, 2:4], axis=-1) / FINGERTIP_WEIGHT))
    tip_above_wrist = float(-((f0[both, :, 3] / FINGERTIP_WEIGHT) - (f0[both, :, 1] / WRIST_WEIGHT)).mean())
    return tgap, tip_above_wrist


def _shape_trajectory_distance(recording, reference):
    # A signer may perform a two-handed gesture higher or lower. Align only the shared
    # height, keeping horizontal placement, the spacing of the hands, and motion direction.
    sample, seen, penalties = recording
    ref_sample, ref_seen, ref_penalties = reference
    sample_height = sample[..., 1].sum(axis=(1, 2)) / np.maximum(1, seen.sum(axis=(1, 2)))
    ref_height = ref_sample[..., 1].sum(axis=(1, 2)) / np.maximum(1, ref_seen.sum(axis=(1, 2)))
    max_shift = MAX_SIGNING_HEIGHT_SHIFT * WRIST_WEIGHT
    distances = []
    for target, target_seen, penalty, height in zip(ref_sample, ref_seen, ref_penalties, ref_height):
        shift = np.clip(height - sample_height, -max_shift, max_shift)[:, None, None] * seen
        aligned = sample.copy()
        aligned[..., 1] += shift
        aligned[..., 3] += shift * (FINGERTIP_WEIGHT / WRIST_WEIGHT)
        distances.append(float(_variant_distances(
            (aligned, seen, penalties), (target[None], target_seen[None], np.asarray([penalty])),
        ).min()))
    return min(distances)


def _bilateral_shape_match(frames, aspect_ratio, recording, bucket, references, *, allowed_ids=None, is_rare_tier=False, trace=None):
    """Sign-agnostic inter-hand relative trajectory & shape matcher for 2-hand signs."""
    f, o, p = recording
    paired = o[0].all(axis=1)
    paired_n = int(paired.sum())
    sample_tgap, sample_up = _bilateral_tip_gap(f, o)
    clasped_upright = (
        not is_rare_tier
        and paired_n >= 7
        and sample_tgap is not None
        and sample_tgap <= 0.22
        and sample_up >= 0.18
        and lost_hand_during_interaction(frames, aspect_ratio)
    )
    if paired_n < 12 and not clasped_upright:
        if trace is not None:
            trace["bilateral_shape"] = {"skipped": "paired_frames_lt_12", "paired_frames": paired_n}
        return None
    sample = (_relative_hand_features(f, o), o, p)
    sample_duplicates = _duplicate_hand_fraction(f, o)

    def adapt(reference):
        feat, observed, penalties = reference
        duplicates = max(0.0, _duplicate_hand_fraction(feat, observed) - sample_duplicates)
        gap_penalty = 0.0
        if sample_tgap is not None and sample_tgap <= 0.22:
            ref_tgap, _ = _bilateral_tip_gap(feat, observed)
            if ref_tgap is not None and ref_tgap > sample_tgap:
                gap_penalty = 0.25 * (ref_tgap - sample_tgap)
        return _relative_hand_features(feat, observed), observed, penalties + 0.20 * duplicates + gap_penalty

    candidates = dict(references)
    if bucket is not None and "relative_coarse" in bucket:
        candidates.update(_indexed_references(sample, bucket, coarse_key="relative_coarse", allowed_ids=allowed_ids))
    if allowed_ids is not None:
        candidates = {key: ref for key, ref in candidates.items() if key in allowed_ids}
    scores = _score_templates(
        sample,
        {key: adapt(ref) for key, ref in candidates.items() if ref[0].shape[2] == 2},
        phase_mode="finger_end",
    )

    ranked = sorted((float(distances.min()), entry_id) for entry_id, distances in scores.items())
    shape_trace = None
    if trace is not None:
        shape_trace = {
            "paired_frames": paired_n,
            "paired_ratio": round(float(paired.mean()), 3),
            "sample_duplicates": round(float(sample_duplicates), 3),
            "max_distance": MAX_HAND_SHAPE_DISTANCE,
            "top_candidates": [
                {"entry_id": eid, "distance": round(dist, 4), "similarity": round(max(0.0, 1.0 - dist), 3)}
                for dist, eid in ranked[:5]
            ],
            "gap_top1_top2": round(ranked[1][0] - ranked[0][0], 4) if len(ranked) > 1 else None,
        }
        trace["bilateral_shape"] = shape_trace

    max_shape_dist = 0.29 if is_rare_tier else MAX_HAND_SHAPE_DISTANCE
    if paired_n < 15 and not clasped_upright:
        max_shape_dist = min(max_shape_dist, 0.24)
    if not ranked or ranked[0][0] > max_shape_dist:
        if shape_trace is not None:
            shape_trace["verdict"] = "out_of_range"
        return None

    distance, entry_id = ranked[0]
    best_ref = candidates.get(entry_id)
    if clasped_upright and paired_n < 12 and best_ref is not None:
        best_ref_tgap, _ = _bilateral_tip_gap(best_ref[0], best_ref[1])
        if best_ref_tgap is not None and best_ref_tgap > 0.26:
            if shape_trace is not None:
                shape_trace["verdict"] = "clasped_low_paired_wide_template"
            return None

    separation = MIN_HAND_SHAPE_SEPARATION
    if distance <= 0.30 and paired.mean() >= 0.80 and sample_duplicates <= 0.05:
        separation = 0.032
    elif clasped_upright or (sample_tgap is not None and sample_tgap <= 0.16 and sample_up >= 0.22):
        separation = 0.020
    if shape_trace is not None:
        shape_trace["separation_required"] = separation
    if len(ranked) > 1 and ranked[1][0] - distance < separation:
        if shape_trace is not None:
            shape_trace["verdict"] = "ambiguous"
        return None

    first = scores[entry_id]
    rivals = [value for key, value in scores.items() if key != entry_id]
    runner_up = np.min(rivals, axis=0) if rivals else np.full_like(first, np.inf)
    agreed = (first <= max_shape_dist) & (runner_up - first >= separation)
    if is_rare_tier and agreed.sum() < 2:
        if shape_trace is not None:
            shape_trace["verdict"] = "rare_shape_uncorroborated"
            shape_trace["agreed_trims"] = int(agreed.sum())
        return None
    full_distance = _shape_trajectory_distance(recording, best_ref)
    if shape_trace is not None:
        shape_trace["trajectory_distance"] = round(full_distance, 4)
    if full_distance > MAX_BILATERAL_SHAPE_TRAJECTORY_DISTANCE:
        if shape_trace is not None:
            shape_trace["verdict"] = "trajectory_out_of_range"
        return None
    method = "hand_shape" if agreed.sum() >= 2 else "trajectory"
    if shape_trace is not None:
        shape_trace["verdict"] = "matched"
        shape_trace["agreed_trims"] = int(agreed.sum())
        shape_trace["match_method"] = method
    return _match_result(distance, entry_id, method)


def _unilateral_contour_match(recording, bucket, references, *, allowed_ids=None, is_rare_tier=False):
    """Sign-agnostic contour + velocity + 5-finger configuration matcher for 1-hand signs across camera framings."""
    f, o, p = recording
    seen0 = o[0, :, 0]
    if int(seen0.sum()) < 16:
        return None
    # Require clear contour excursion so micro-twitches do not match via lateral offset attenuation
    contour_xy = f[0, seen0, 0, 4:8] / CONTOUR_WEIGHT
    excursion = float(max(np.linalg.norm(np.ptp(contour_xy[:, :2], axis=0)), np.linalg.norm(np.ptp(contour_xy[:, 2:4], axis=0))))
    if excursion < 0.16:
        return None

    sample = (_unilateral_contour_features(f, o), o, p)
    candidates = dict(references)
    if bucket is not None and "contour_coarse" in bucket:
        candidates.update(_indexed_references(sample, bucket, coarse_key="contour_coarse", allowed_ids=allowed_ids))
    if allowed_ids is not None:
        candidates = {key: ref for key, ref in candidates.items() if key in allowed_ids}
    adapted = {
        key: (_unilateral_contour_features(ref[0], ref[1]), ref[1], ref[2])
        for key, ref in candidates.items()
        if ref[0].shape[2] == 1
    }
    scores = _score_templates(sample, adapted)
    ranked = sorted((float(distances.min()), entry_id) for entry_id, distances in scores.items())
    max_dist = 0.25 if is_rare_tier else MAX_UNI_CONTOUR_DISTANCE
    if not ranked or ranked[0][0] > max_dist:
        return None
    distance, entry_id = ranked[0]
    sep = 0.026 if distance <= 0.22 else MIN_UNI_CONTOUR_SEPARATION
    if len(ranked) > 1 and ranked[1][0] - distance < sep:
        return None
    first = scores[entry_id]
    rivals = [value for key, value in scores.items() if key != entry_id]
    runner_up = np.min(rivals, axis=0) if rivals else np.full_like(first, np.inf)
    if is_rare_tier and int(((first <= max_dist) & (runner_up - first >= sep)).sum()) < 2:
        return None
    if float(_variant_distances(recording, candidates[entry_id]).min()) > MAX_UNILATERAL_SHAPE_TRAJECTORY_DISTANCE:
        return None
    return _match_result(distance, entry_id, method="hand_shape")


def _signing_hand_observations(frames, aspect_ratio):
    """Return real hand observations within signing space, excluding hands at rest."""
    frames = np.asarray(frames, dtype=np.float32).copy()
    measured = _tracked_hands(frames)
    frames[..., 1] /= aspect_ratio
    normalized = normalize_sequence_tjc_v2(frames)
    return measured & (normalized[:, [33, 54], 1] <= 0.90)


def _unilateral_core_rank(frames, aspect_ratio, recording, bucket, allowed_ids):
    """Rank a well-observed single hand with weaker position and speed assumptions."""
    if bucket is None:
        return [], {}
    measured = _signing_hand_observations(frames, aspect_ratio)
    dominant = int(measured.sum(axis=0).max())
    if dominant < 10 or measured.all(axis=1).sum() > max(3, 0.15 * dominant):
        return [], {}
    features, observed, penalties = recording
    sample = (_unilateral_core_features(features, observed), observed, penalties)
    references = dict(_indexed_references(sample, bucket, coarse_key="core_coarse", allowed_ids=allowed_ids))
    scaled = {
        key: (_unilateral_core_features(feat, seen), seen, cut)
        for key, (feat, seen, cut) in references.items()
    }
    scores = _score_templates(sample, scaled, phase_mode="none")
    return sorted((float(score.min()), key) for key, score in scores.items()), references


def _with_unilateral_core(result, frames, aspect_ratio, recording, bucket, allowed_ids, completed, candidate_scores, trace):
    if not completed:
        return result
    if result and result["status"] == "word_matched":
        if result["match_method"] == "hand_shape":
            # Contour alone can match the common first movement of two signs.
            # Require independent trajectory and finger evidence to commit it.
            result = {**result, "requires_user_confirmation": True}
        elif result["similarity"] >= 0.78:
            return result
    ranked, references = _unilateral_core_rank(frames, aspect_ratio, recording, bucket, allowed_ids)
    if not ranked:
        return result
    distance, entry_id = ranked[0]
    margin = ranked[1][0] - distance if len(ranked) > 1 else np.inf
    path_distance = float(_variant_distances(recording, references[entry_id]).min())
    if candidate_scores is not None:
        # A near miss can still be useful for manual selection when the relative
        # shape and the absolute trajectory both support the same candidate.
        hint_limit = MAX_SINGLE_HAND_CORE_HINT_DISTANCE
        if margin >= 0.012 and path_distance <= 0.68:
            hint_limit = MAX_CORROBORATED_CORE_HINT_DISTANCE
        candidate_scores.extend({
            "entry_id": key, "similarity": round(1.0 - score, 3), "distance": round(score, 4),
            "match_method": "unilateral_core",
        } for score, key in ranked[:5]
            if score <= MAX_SINGLE_HAND_CORE_HINT_DISTANCE or (key == entry_id and score <= hint_limit))
    if trace is not None:
        trace["unilateral_core"] = {
            "entry_id": entry_id, "distance": round(distance, 4),
            "margin": round(margin, 4) if np.isfinite(margin) else None,
        }
    if (distance > MAX_SINGLE_HAND_CORE_DISTANCE or margin < MIN_SINGLE_HAND_CORE_SEPARATION or
            (result and result["status"] == "word_matched" and result["entry_id"] != entry_id)):
        return result
    # Keep corroboration of absolute path and its beginning/end for auto commit.
    if path_distance > 0.47:
        return result
    if trace is not None:
        trace["verdict"] = "unilateral_core_matched"
    return _match_result(distance, entry_id, method="unilateral_core")


def _with_personal_examples(result, recording, examples, candidate_scores, trace):
    if not examples:
        return result
    hands = recording[0].shape[2]
    ranked = []
    for entry_id, samples in examples.items():
        sample_scores = sorted(
            float(_variant_distances(recording, ref).min())
            for ref in samples if ref[0].shape[2] == hands
        )
        if sample_scores:
            ranked.append((sample_scores[0], entry_id, sum(score <= 0.22 for score in sample_scores)))
    ranked.sort()
    if not ranked or ranked[0][0] > 0.40:
        return result
    if candidate_scores is not None:
        candidate_scores.extend(
            _match_result(distance, entry_id, method="personal_template")
            for distance, entry_id, _ in ranked[:3] if distance <= 0.40
        )
    distance, entry_id, support = ranked[0]
    margin = ranked[1][0] - distance if len(ranked) > 1 else np.inf
    if trace is not None:
        trace["personal"] = {"entry_id": entry_id, "distance": round(distance, 4), "supporting_samples": support}
    if distance > 0.22 or margin < 0.075:
        return result
    if (result and result["status"] == "word_matched" and result["entry_id"] != entry_id
            and result["similarity"] >= 0.85 and not result.get("requires_user_confirmation")):
        return result
    if result and result["status"] == "word_matched" and result["entry_id"] == entry_id:
        return result
    matched = _match_result(distance, entry_id, method="personal_template")
    matched["requires_user_confirmation"] = support < 2
    if trace is not None:
        trace["verdict"] = "personal_template_matched"
    return matched


def corroborate_sstcn_entry(frames, aspect_ratio, entry_id):
    """Generic QIPEDC template corroboration for SSTCN predictions without any per-sign hardcoding."""
    if not entry_id:
        return False
    idx = _active_index()
    if idx is None or entry_id not in idx["id_set"]:
        return False

    recording = _variants(frames, aspect_ratio)
    if recording is None:
        return False
    recording = _index_feature_view(recording, idx)
    hands = recording[0].shape[2]
    bucket = idx["buckets"].get(hands)
    if bucket is None or entry_id not in bucket["ids"]:
        return False
    pos = bucket["ids"].index(entry_id)
    st, en = bucket["slices"][pos]
    ref_f = bucket["features"][st:en].astype(np.float32, copy=False)
    ref_o = bucket["observed"][st:en]
    ref_p = bucket["penalties"][st:en]

    # Generic vertical signing-zone check: if the template reaches head/face level, a chest-only clip cannot corroborate it.
    sample_o = recording[1][0]
    if sample_o.any() and ref_o[0].any():
        sample_min_tip_y = float((recording[0][0, :, :, 3][sample_o] / FINGERTIP_WEIGHT).min())
        ref_min_tip_y = float((ref_f[0, :, :, 3][ref_o[0]] / FINGERTIP_WEIGHT).min())
        if sample_min_tip_y - ref_min_tip_y > 0.22:
            return False

    min_dist = float(_variant_distances(recording, (ref_f, ref_o, ref_p)).min())
    return min_dist <= (0.44 if hands == 1 else 0.54)


def _effective_unilateral_separation(best_distance):
    if best_distance <= 0.18:
        return 0.015
    if best_distance <= 0.24:
        return 0.026
    return MIN_SEPARATION


def _collect_extra_references(entries, references, allowed_set, idx, *, include_indexed=False):
    for entry_id, video_id in entries or ():
        if allowed_set is not None and entry_id not in allowed_set:
            continue
        if entry_id in references:
            continue
        if not include_indexed and idx is not None and entry_id in idx["id_set"]:
            continue
        if not verify_reference_video_id(video_id)["eligible"]:
            continue
        path = reference_path(video_id)
        if not path.is_file():
            continue
        reference = _reference_variants(path)
        if reference is not None:
            references[entry_id] = _index_feature_view(reference, idx)


def _evaluate_tier(
    frames,
    aspect_ratio,
    recording,
    bucket,
    entries,
    idx,
    *,
    allowed_set=None,
    is_rare_tier=False,
    trace=None,
):
    hands = recording[0].shape[2]
    bilateral = hands == 2
    references = dict(_indexed_references(recording, bucket, allowed_ids=allowed_set)) if bucket is not None else {}
    _collect_extra_references(entries, references, allowed_set, idx, include_indexed=False)
    if bucket is not None and idx is not None and entries:
        main_bucket = idx["buckets"][hands]
        if main_bucket is not bucket:
            id_to_pos = None
            for entry_id, _ in entries:
                if entry_id in references or (allowed_set is not None and entry_id not in allowed_set):
                    continue
                if id_to_pos is None:
                    id_to_pos = {eid: pos for pos, eid in enumerate(main_bucket["ids"])}
                pos = id_to_pos.get(entry_id)
                if pos is not None:
                    st, en = main_bucket["slices"][pos]
                    references[entry_id] = (
                        main_bucket["features"][st:en].astype(np.float32, copy=False),
                        main_bucket["observed"][st:en],
                        main_bucket["penalties"][st:en],
                    )

    ranked = sorted((float(distances.min()), entry_id) for entry_id, distances in _score_templates(recording, references).items())
    if bilateral:
        max_distance = MAX_RARE_BILATERAL_DISTANCE if is_rare_tier else MAX_BILATERAL_DISTANCE
        min_separation = MIN_BILATERAL_SEPARATION if (not ranked or ranked[0][0] > 0.24) else 0.035
    else:
        max_distance = MAX_RARE_DISTANCE if is_rare_tier else MAX_DISTANCE
        min_separation = _effective_unilateral_separation(ranked[0][0]) if ranked else MIN_SEPARATION
    in_range = bool(ranked and ranked[0][0] <= max_distance)
    gap = round(ranked[1][0] - ranked[0][0], 4) if len(ranked) > 1 else None
    tier_info = {
        "tier": "full_3240" if is_rare_tier else "priority_common",
        "bucket_hands": int(hands),
        "bucket_total_words": len(bucket["ids"]) if bucket is not None else 0,
        "scored_candidates": len(ranked),
        "allowed_filter_size": len(allowed_set) if allowed_set is not None else None,
        "max_distance": max_distance,
        "min_separation": min_separation,
        "in_range": in_range,
        "gap_top1_top2": gap,
        "top_candidates": [
            {"entry_id": eid, "distance": round(dist, 4), "similarity": round(max(0.0, 1.0 - dist), 3)}
            for dist, eid in ranked[:5]
        ],
    }

    if in_range and (len(ranked) == 1 or ranked[1][0] - ranked[0][0] >= min_separation):
        paired_n = int(recording[1][0].all(axis=1).sum()) if bilateral else 24
        weak_overlap = False
        if bilateral and (is_rare_tier or ranked[0][0] > 0.35):
            filled_clip = _fill_short_gaps(np.asarray(frames, dtype=np.float32), allow_extended_pose_bridge=True)
            obs_counts = _tracked_hands(filled_clip).sum(axis=0)
            weak_overlap = paired_n < 10 or int(obs_counts.min()) < 10
        elif not bilateral and is_rare_tier and ranked[0][0] > 0.22:
            raw_obs = _tracked_hands(np.asarray(frames, dtype=np.float32)).sum(axis=0)
            weak_overlap = int(raw_obs.max()) < 12
        if weak_overlap:
            tier_info["verdict"] = "bilateral_too_few_paired_frames" if bilateral else "unilateral_rare_too_short"
        else:
            tier_info["verdict"] = "trajectory_matched"
            if trace is not None:
                trace.update(tier_info)
            return _match_result(*ranked[0]), tier_info

    if not bilateral:
        if (uni_shape := _unilateral_contour_match(
            recording, bucket, references, allowed_ids=allowed_set, is_rare_tier=is_rare_tier
        )) is not None:
            tier_info["verdict"] = "unilateral_contour_matched"
            if trace is not None:
                trace.update(tier_info)
            return uni_shape, tier_info

    run_shape = bilateral and (not is_rare_tier or (ranked and ranked[0][0] <= 0.45))
    if run_shape:
        shape_trace = {} if trace is not None else None
        if (recovered := _bilateral_shape_match(
            frames, aspect_ratio, recording, bucket, references,
            allowed_ids=allowed_set, is_rare_tier=is_rare_tier, trace=shape_trace,
        )):
            tier_info["verdict"] = "bilateral_shape_matched"
            if shape_trace:
                tier_info.update(shape_trace)
            if trace is not None:
                trace.update(tier_info)
            return recovered, tier_info
        if shape_trace:
            tier_info.update(shape_trace)

    if in_range:
        tier_info["verdict"] = "ambiguous_word"
        return {
            "motion_detected": True,
            "status": "ambiguous_word",
            "suggestions": tier_info["top_candidates"][:3],
        }, tier_info
    tier_info["verdict"] = "out_of_range" if ranked else "no_candidates"
    return None, tier_info


def match_live_words(
    frames, aspect_ratio, entries, *, allowed_ids=None, vocab_scope="all", trace=None, candidate_scores=None,
    completed=False, personal_examples=None,
):
    """Two-tier generic QIPEDC kinematic & hand-configuration matcher."""
    allowed_set = frozenset(allowed_ids) if allowed_ids is not None else None
    if allowed_set is not None and not allowed_set:
        if trace is not None:
            trace["verdict"] = "empty_allowed_ids"
        return None
    if (recording := _variants(frames, aspect_ratio, trace=trace)) is None:
        if trace is not None:
            trace["verdict"] = "variants_rejected"
        return None
    hands = recording[0].shape[2]
    idx = _active_index()
    personal_recording = recording
    recording = _index_feature_view(recording, idx)
    full_bucket = idx["buckets"][hands] if idx is not None else None
    priority_bucket = full_bucket.get("priority_sub") if full_bucket is not None else None

    def finish(result, *, bucket=full_bucket):
        if hands == 1:
            result = _with_unilateral_core(
                result, frames, aspect_ratio, recording, bucket, allowed_set, completed, candidate_scores, trace,
            )
        result = _with_personal_examples(result, personal_recording, personal_examples, candidate_scores, trace)
        if result and result["status"] == "word_matched":
            result["hand_count"] = hands
        return result

    if priority_bucket is None or (allowed_set is not None and len(allowed_set) <= 40):
        res, info = _evaluate_tier(
            frames, aspect_ratio, recording, full_bucket, entries, idx,
            allowed_set=allowed_set, is_rare_tier=False, trace=trace,
        )
        if candidate_scores is not None:
            candidate_scores.extend(info["top_candidates"])
        if trace is not None and (res is None or res.get("status") == "ambiguous_word"):
            trace.update(info)
        return finish(res)

    res_t1, info_t1 = _evaluate_tier(
        frames, aspect_ratio, recording, priority_bucket, entries, idx,
        allowed_set=allowed_set, is_rare_tier=False, trace=trace,
    )
    if candidate_scores is not None:
        candidate_scores.extend(info_t1["top_candidates"])
    t1_matched = bool(res_t1 and res_t1.get("status") == "word_matched")
    t1_dist = (
        float(info_t1["top_candidates"][0]["distance"])
        if info_t1.get("top_candidates")
        else (1.0 - float(res_t1.get("similarity", 0.0)) if t1_matched else np.inf)
    )
    if t1_matched and (vocab_scope == "common" or t1_dist <= 0.14):
        return finish(res_t1, bucket=priority_bucket if vocab_scope == "common" else full_bucket)

    if vocab_scope == "common":
        if trace is not None:
            trace.update(info_t1)
        return finish(res_t1, bucket=priority_bucket)

    res_t2, info_t2 = _evaluate_tier(
        frames, aspect_ratio, recording, full_bucket, entries, idx,
        allowed_set=allowed_set, is_rare_tier=True, trace=trace,
    )
    if candidate_scores is not None:
        candidate_scores.extend(info_t2["top_candidates"])
    if res_t2 and res_t2.get("status") == "word_matched":
        t2_dist = (
            float(info_t2["top_candidates"][0]["distance"])
            if info_t2.get("top_candidates")
            else (1.0 - float(res_t2.get("similarity", 0.0)))
        )
        if not t1_matched or t2_dist <= t1_dist - 0.05:
            return finish(res_t2)
    if t1_matched:
        if trace is not None:
            trace.update(info_t1)
        return finish(res_t1)

    chosen_res = res_t2 if res_t2 is not None else res_t1
    chosen_info = info_t2 if res_t2 is not None else (info_t1 if res_t1 is not None else info_t2)
    if trace is not None:
        trace.update(chosen_info)
        trace["tier1_top1"] = (info_t1.get("top_candidates") or [None])[0]
    return finish(chosen_res)
