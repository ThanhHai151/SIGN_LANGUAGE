"""Compare webcam motion with a library of prepared QIPEDC references."""

from functools import lru_cache
import os
from pathlib import Path
import threading

import numpy as np

from backend.models import video_cache
from backend.vision.qipedc_practice import (
    TIME_SIGN_ENTRY,
    TIME_SIGN_VIDEO,
    _fill_short_gaps,
    _tracked_hands,
    _visible_clip,
    prepare_reference,
    reference_path,
)
from backend.vision.skeleton_norm import normalize_sequence_tjc_v2
from backend.vision.word_scope import verify_reference_video_id


DEFAULT_WORDS = (
    "q-an-du", "q-an-vua", "q-an-it", "q-com-2", "q-nem", "q-lau-mieng", "q-thom",
    "q-nan-ni", "q-cau-nguyen", "q-vai-lay",
    "q-xin-loi", "q-ky-nang", "q-kha-nang", "q-khong-co",
)
BUILTIN_WORDS = (TIME_SIGN_ENTRY, *DEFAULT_WORDS)
_CHUNG_TOI_IDS = (
    "q-chung-toi-dai-tu",
    "q-chung-toi-2-nguoi-dai-tu",
    "q-chung-toi-3-nguoi-dai-tu",
    "q-chung-toi-4-nguoi-dai-tu",
    "q-chung-toi-5-nguoi-dai-tu",
)
MAX_EXTRA_WORDS = 12
FRAMES = 24
COARSE_BINS = 6
STAGE1_TOP_K = 18
DTW_REFERENCE_BATCH_SIZE = 6
MAX_DISTANCE = 0.35
MAX_RARE_DISTANCE = 0.28
MIN_SEPARATION = 0.05
MAX_BILATERAL_DISTANCE = 0.52
MAX_RARE_BILATERAL_DISTANCE = 0.44
MIN_BILATERAL_SEPARATION = 0.08
MAX_HAND_SHAPE_DISTANCE = 0.35
MIN_HAND_SHAPE_SEPARATION = 0.05
HAND_MISMATCH_PENALTY = 0.10
RAISED_WRIST_MAX_Y = 0.68
REFLECT_X = np.tile([-1, 1], 7).astype(np.float32)
HAND_SHAPE_WEIGHTS = np.repeat(np.array([0.6, 0.3, 2.0], dtype=np.float32), [2, 2, 10])
WRIST_WEIGHT = 0.60
FINGERTIP_WEIGHT = 0.95
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

    # The pose wrists can corroborate that both arms remain raised when the
    # hand model loses finger landmarks. They do not stand in for hidden fingers.
    shoulder_y = clip[:, [11, 12], 1].mean(axis=1)
    pose_wrists = clip[:, [15, 16], :]
    raised = np.any(pose_wrists != 0, axis=2).all(axis=1) & (
        pose_wrists[..., 1] < (shoulder_y + 0.80 * shoulder_width * aspect_ratio)[:, None]
    ).all(axis=1)
    # A release ends the interaction so a later single-hand sign can be read normally.
    released = ~raised
    breaks = np.flatnonzero(released[:-2] & released[1:-1] & released[2:])
    start = int(breaks[-1] + 3) if breaks.size else 0
    if start > end or close_pair[start:].sum() < 2:
        return False

    first_pair = start + int(np.flatnonzero(close_pair[start:])[0])
    tracked = hands[first_pair:]
    visible = tracked.any(axis=1)
    return bool(visible.sum() >= 8 and tracked[visible].all(axis=1).mean() < 0.60)


def _variants(raw, aspect_ratio, *, reference=False, trace=None):
    try:
        clip, _ = _visible_clip(raw)
    except ValueError as exc:
        if trace is not None:
            trace["variants_reject_reason"] = f"visible_clip_failed: {exc}"
        return None

    clip = _fill_short_gaps(clip)
    clip[..., 1] /= aspect_ratio
    observed = _tracked_hands(clip)
    body = normalize_sequence_tjc_v2(clip)
    if reference:
        # Studio QIPEDC videos include waist-level arm raises/drops that stay outside webcam framing.
        raised = observed & (body[:, [33, 54], 1] <= RAISED_WRIST_MAX_Y)
        raised_idx = np.flatnonzero(raised.any(axis=1))
        if len(raised_idx) >= 7:
            sl = slice(int(raised_idx[0]), int(raised_idx[-1]) + 1)
            clip, observed, body = clip[sl], observed[sl], body[sl]
    hands = body[:, 33:].reshape(len(body), 2, 21, 3)
    movement = np.zeros(2)
    wrist_y_med = np.full(2, np.inf)
    for side in (0, 1):
        points = hands[observed[:, side], side, :, :2]
        if len(points) >= 1:
            wrist_y_med[side] = float(np.median(points[:, 0, 1]))
        if len(points) >= 5:
            low, high = np.percentile(points, [5, 95] if not reference else [10, 90], axis=0)
            movement[side] = np.linalg.norm(high - low, axis=-1).mean()

    # Allow smaller chest/face-level movements (>=0.065) on live webcam when wrist is clearly raised;
    # keep calibrated 0.075 baseline otherwise.
    min_move = np.array([
        0.065 if (not reference and wrist_y_med[side] <= 0.55) else 0.075
        for side in (0, 1)
    ], dtype=np.float32)
    active = (movement >= min_move) & (movement >= 0.23 * max(movement.max(), 1e-6))
    obs_counts = observed.sum(axis=0)
    forced_bilateral = False
    if not active.all() and (obs_counts >= 2).all():
        if reference:
            # Retain bilateral structure when studio reference has one occluded raised hand (e.g. "năn nỉ"),
            # without turning 1-hand signs whose idle hand hangs down at the waist into 2-hand templates.
            if float(wrist_y_med.max()) <= 0.82:
                forced_bilateral = True
                active[:] = True
        else:
            # On live webcam, avoid letting 2-5 stray frames or a waist-level resting hand poison a 1-hand sign.
            # Only force bilateral when the weaker hand is raised (<= RAISED_WRIST_MAX_Y) with sustained presence
            # OR genuine close interaction/clasp (>=4 frames with wrists within 0.72 shoulder widths).
            min_obs = int(obs_counts.min())
            sustained = (
                min_obs >= max(6, int(0.24 * len(clip)))
                and float(wrist_y_med.max()) <= RAISED_WRIST_MAX_Y
            )
            paired_mask = observed.all(axis=1)
            close_clasp = False
            if min_obs >= 4 and paired_mask.sum() >= 3:
                wrist_gap = np.linalg.norm(hands[paired_mask, 0, 0, :2] - hands[paired_mask, 1, 0, :2], axis=-1)
                close_clasp = bool((wrist_gap <= 0.72).sum() >= 3)
            if sustained or close_clasp:
                forced_bilateral = True
                active[:] = True
    if trace is not None:
        trace.update({
            "left_movement": round(float(movement[0]), 4),
            "right_movement": round(float(movement[1]), 4),
            "movement_threshold": float(min_move.min()),
            "observed_counts": [int(obs_counts[0]), int(obs_counts[1])],
            "active_hands": "both" if active.all() else ("left" if active[0] else ("right" if active[1] else "none")),
            "bilateral_forced_by_2frame_rule": forced_bilateral,
        })
    observed = observed[:, active]
    tracked = np.flatnonzero(observed.any(axis=1))
    coverage = float(observed[tracked[0]:tracked[-1] + 1].any(axis=1).mean()) if len(tracked) else 0.0
    if len(tracked) < 7 or coverage < 0.70:
        if trace is not None:
            trace.update({
                "tracked_frames": int(len(tracked)),
                "tracking_coverage": round(coverage, 3),
                "variants_reject_reason": (
                    "no_active_hand_below_0.09" if not active.any()
                    else ("too_few_tracked_frames_<7" if len(tracked) < 7 else "low_tracking_coverage_<0.70")
                ),
            })
        return None
    bilateral = active.all()

    # Use the mouth as a landmark on the upper body and use palm size for hand shape.
    # In image coordinates y is scaled to the width before normalization.
    mouth = body[:, [9, 10], :2].mean(axis=1)
    hands = hands[:, active]
    tips = hands[:, :, [4, 8, 12, 16, 20], :2]
    wrist = hands[:, :, 0, :2]
    palm_size = np.maximum(0.04, np.linalg.norm(hands[:, :, 9] - hands[:, :, 0], axis=-1))
    shape = (tips - wrist[:, :, None]) / palm_size[:, :, None, None]
    features = np.concatenate((
        WRIST_WEIGHT * (wrist - mouth[:, None]),
        FINGERTIP_WEIGHT * (tips.mean(axis=2) - mouth[:, None]),
        0.22 * shape.reshape(len(clip), -1, 10),
    ), axis=-1)
    if not bilateral and active[0]:
        features *= REFLECT_X

    usable = slice(tracked[0], tracked[-1] + 1) if bilateral else tracked
    features, observed = features[usable], observed[usable]
    cuts = sorted({0, len(features) // 6})
    trims = [(start, end) for start in cuts for end in cuts]
    penalty_list = [0.025 if start or end else 0.0 for start, end in trims]
    active_trim = None
    if not reference:
        active_trim = _active_motion_trim(features, observed)
        if active_trim is not None and active_trim not in trims:
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
    if peak < 0.025:
        return None
    active = np.flatnonzero(speed >= max(0.020, 0.14 * peak))
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
    return video_id == TIME_SIGN_VIDEO or _reference_variants(reference_path(video_id)) is not None


def _variant_distances(recording, reference):
    sample, sample_hands, cut = recording
    template, template_hands, ref_cut = reference
    n_sample, sample_frames, hands, feature_count = sample.shape
    n_template, template_frames, _, _ = template.shape
    # Compute squared Euclidean distances without allocating a feature vector for each frame pair.
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
    # Score all short endpoint trims together. Restrict the temporal warp
    # so that matching a single pose cannot explain the entire trajectory.
    dp = np.full((FRAMES + 1, FRAMES + 1, len(sample), len(template)), np.inf, dtype=np.float32)
    dp[0, 0] = 0
    for i, j in WARP_DIAGONALS:
        dt = cost[:, :, i - 1, j - 1].transpose(2, 0, 1)
        dp[i, j] = np.minimum(dp[i - 1, j - 1] + dt, np.minimum(dp[i - 1, j], dp[i, j - 1])) + dt
    return dp[-1, -1] / (2 * FRAMES) + cut[:, None] + ref_cut


def _score_templates(recording, references):
    hands = recording[0].shape[2]
    usable = []
    for entry, ref in references.items():
        if ref[0].shape[2] == hands:
            usable.append((entry, ref))
        elif hands == 1 and ref[0].shape[2] == 2 and entry in ("q-khong-co",):
            # Cử chỉ đối xứng hai tay có biến thể thực tế một tay
            # Trích xuất tay phải (index 1) và tay trái (index 0) tạo thành template 1 tay
            ref_r = (ref[0][:, :, 1:2, :], ref[1][:, :, 1:2], ref[2])
            ref_l = (ref[0][:, :, :1, :], ref[1][:, :, :1], ref[2])
            usable.append((entry, ref_r))
            usable.append((entry, ref_l))
    scores = {}
    for offset in range(0, len(usable), DTW_REFERENCE_BATCH_SIZE):
        batch = usable[offset:offset + DTW_REFERENCE_BATCH_SIZE]
        ids, templates = zip(*batch)
        combined = tuple(np.concatenate(parts, axis=0) for parts in zip(*templates))
        first_variant = np.cumsum([0, *(len(ref[0]) for ref in templates[:-1])])
        distances = np.minimum.reduceat(_variant_distances(recording, combined), first_variant, axis=1)
        for entry_id, dist in zip(ids, distances.T):
            if entry_id not in scores or dist.min() < scores[entry_id].min():
                scores[entry_id] = dist
    return scores


def _relative_hand_features(features, observed):
    """Emphasize how the wrists and finger shapes differ between the two hands."""
    seen = observed[..., None]
    center = (features * seen).sum(axis=2, keepdims=True) / np.maximum(1, seen.sum(axis=2, keepdims=True))
    return (features - 0.8 * center.astype(features.dtype)) * HAND_SHAPE_WEIGHTS


def _duplicate_hand_fraction(features, observed):
    # Holistic can return the same outline as both hand tracks when palms overlap.
    gap = np.linalg.norm(features[:, :, 0] - features[:, :, 1], axis=-1)
    return float((observed.all(axis=2) & (gap < 0.09)).mean())


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


def _relative_coarse_chunked(features, observed, chunk_size=1024):
    """Compute relative_coarse in small chunks to avoid a ~200 MB temporary RAM spike on Render Free."""
    total, _, hands, dims = features.shape
    out = np.empty((total, COARSE_BINS, hands, dims), dtype=np.float32)
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        f_chunk = features[start:end].astype(np.float32, copy=False)
        o_chunk = observed[start:end]
        out[start:end] = _coarse_pool(_relative_hand_features(f_chunk, o_chunk), o_chunk)
    return out


def _slice_sub_bucket(bucket, keep_set):
    """Build a compact pre-sliced bucket view for fast Stage-1 priority search (~4x fewer variants)."""
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
    if "relative_coarse" in bucket:
        sub["relative_coarse"] = bucket["relative_coarse"][v_idx]
        sub["relative_coarse_squared"] = bucket["relative_coarse_squared"][v_idx]
    return sub


@lru_cache(maxsize=1)
def _default_priority_ids():
    from backend.models.lexicon import load_lexicon

    lex = load_lexicon()
    return frozenset(
        eid for eid, entry in lex["by_id"].items() if entry.get("common")
    ) | frozenset(BUILTIN_WORDS)


def _project_idle_waist_bilateral_to_uni(uni_bucket, bi_bucket):
    """Rescue 1-hand studio signs in bi_bucket whose idle hand merely hung down at the hip/waist."""
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

    target_ids = frozenset(_CHUNG_TOI_IDS) - existing
    for ei, eid in enumerate(bi_bucket["ids"]):
        if eid not in target_ids:
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
                pts = f0[o0[:, s], s, :2].astype(np.float32)
                wy_med[s] = float(np.median(pts[:, 1]))
                w_rng[s] = float(np.linalg.norm(np.ptp(pts, axis=0)))
        act = 0 if wy_med[0] < wy_med[1] else 1
        idle = 1 - act
        if not (obs_n[act] >= 12 and wy_med[act] <= 0.72):
            continue
        if not (obs_n[idle] < 4 or (wy_med[idle] >= 0.92 and w_rng[idle] <= 0.16)):
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


@lru_cache(maxsize=2)
def _load_index_cached(path_str, mtime_ns):
    del mtime_ns
    priority_set = _default_priority_ids()
    with np.load(path_str, allow_pickle=False) as data:
        buckets = {}
        all_ids = []
        for hands, prefix in ((1, "uni"), (2, "bi")):
            ids = [str(x) for x in data[f"{prefix}_ids"].tolist()]
            all_ids.extend(ids)
            # Keep Stage-2 features in compact float16 (~28 MB vs ~57 MB); cast only top-K slices to float32.
            feats = data[f"{prefix}_features"]
            obs = data[f"{prefix}_observed"].astype(bool)
            pens = data[f"{prefix}_penalties"].astype(np.float32)
            v2e = data[f"{prefix}_var_to_entry"].astype(np.int32)
            coarse = data[f"{prefix}_coarse"].astype(np.float32)
            slices = []
            if len(ids):
                boundaries = np.flatnonzero(np.diff(v2e)) + 1
                starts = np.r_[0, boundaries]
                ends = np.r_[boundaries, len(v2e)]
                slices = list(zip(starts.tolist(), ends.tolist()))
            bucket = buckets[hands] = {
                "ids": ids, "features": feats, "observed": obs,
                "penalties": pens, "var_to_entry": v2e, "coarse": coarse, "slices": slices,
            }
            if hands == 2:
                bucket["relative_coarse"] = _relative_coarse_chunked(feats, obs)
        _project_idle_waist_bilateral_to_uni(buckets[1], buckets[2])
        for hands in (1, 2):
            bucket = buckets[hands]
            for key in ("coarse", "relative_coarse"):
                if key in bucket:
                    bucket[key + "_squared"] = np.einsum("vthd,vthd->vth", bucket[key], bucket[key])
            # Pre-slice common+builtin sub-bucket so Tier-1 search skips ~80% of rare variants.
            bucket["priority_sub"] = _slice_sub_bucket(bucket, priority_set)
    return {"buckets": buckets, "all_ids": tuple(all_ids), "id_set": frozenset(all_ids)}


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
    sample_coarse = _coarse_pool(sample_f, sample_o)  # [V_s, 6, H, 14]
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
        tmpl_coarse = bucket[coarse_key]                    # [N_v, 6, H, 14]
        tmpl_squared = bucket[coarse_key + "_squared"]
        tmpl_pen = bucket["penalties"]                    # [N_v]
        num_pool = n_entries

    best_var = np.full(len(tmpl_coarse), np.inf, dtype=np.float32)
    for sv, cut in zip(sample_coarse, sample_cut):
        dot = np.einsum("vthd,thd->vth", tmpl_coarse, sv)
        squared = tmpl_squared + (sv * sv).sum(axis=-1) - 2 * dot
        diff = np.sqrt(np.maximum(squared, 0)).mean(axis=(1, 2)) + tmpl_pen + float(cut)
        np.minimum(best_var, diff, out=best_var)
    entry_best = np.full(num_pool, np.inf, dtype=np.float32)
    np.minimum.at(entry_best, sub_v2e, best_var)
    if slice_allowed:
        chosen = np.argpartition(entry_best, top_k)[:top_k]
        return [allowed_idx[int(k)] for k in chosen]
    if allowed_idx is not None:
        sub = entry_best[allowed_idx]
        chosen = np.argpartition(sub, top_k)[:top_k]
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
    return {"motion_detected": True, "status": "word_matched", "entry_id": entry_id, "match_method": method,
            "similarity": round(1.0 - score, 3)}


def _bilateral_tip_gap(features, observed):
    """Height-invariant median distance between left and right fingertip centroids on paired frames."""
    f0, o0 = features[0], observed[0]
    both = o0.all(axis=1)
    if not both.any():
        return None, 0.0
    tgap = float(np.median(np.linalg.norm(f0[both, 0, 2:4] - f0[both, 1, 2:4], axis=-1) / FINGERTIP_WEIGHT))
    tip_above_wrist = float(-((f0[both, :, 3] / FINGERTIP_WEIGHT) - (f0[both, :, 1] / WRIST_WEIGHT)).mean())
    return tgap, tip_above_wrist


def _bilateral_shape_match(frames, aspect_ratio, recording, bucket, references, *, allowed_ids=None, is_rare_tier=False, trace=None):
    f, o, p = recording
    paired = o[0].all(axis=1)
    paired_n = int(paired.sum())
    sample_tgap, sample_up = _bilateral_tip_gap(f, o)
    # Upright clasped-palm signs (e.g. "năn nỉ") frequently occlude one hand during contact.
    clasped_upright = (
        not is_rare_tier
        and paired_n >= 7
        and sample_tgap is not None
        and sample_tgap <= 0.22
        and sample_up >= 0.18
        and lost_hand_during_interaction(frames, aspect_ratio)
    )
    # Require >=50% (12/24) two-hand frames unless palms are genuinely clasped upright:
    # when only one hand is visible, _relative_hand_features shrinks features by 5x toward zero.
    if paired_n < 12 and not clasped_upright:
        if trace is not None:
            trace["bilateral_shape"] = {"skipped": "paired_frames_lt_12", "paired_frames": paired_n}
        return None
    sample = (_relative_hand_features(f, o), o, p)
    sample_duplicates = _duplicate_hand_fraction(f, o)

    def adapt(reference):
        feat, observed, penalties = reference
        duplicates = max(0, _duplicate_hand_fraction(feat, observed) - sample_duplicates)
        gap_penalty = 0.0
        # When the live hands are clasped/touching (tgap <= 0.22), penalize separated-hand templates
        # (e.g. "công bằng", "lửa") without penalizing natural arm-width variance on wide signs.
        if sample_tgap is not None and sample_tgap <= 0.22:
            ref_tgap, _ = _bilateral_tip_gap(feat, observed)
            if ref_tgap is not None and ref_tgap > sample_tgap:
                gap_penalty = 0.25 * (ref_tgap - sample_tgap)
        return _relative_hand_features(feat, observed), observed, penalties + 0.20 * duplicates + gap_penalty

    candidates = dict(references)
    if bucket is not None:
        candidates.update(_indexed_references(sample, bucket, coarse_key="relative_coarse", allowed_ids=allowed_ids))
    if allowed_ids is not None:
        candidates = {key: ref for key, ref in candidates.items() if key in allowed_ids}
    scores = _score_templates(sample, {key: adapt(ref) for key, ref in candidates.items() if ref[0].shape[2] == 2})

    # A closer view of the public năn nỉ sample preserves fingertips during contact.
    if (allowed_ids is None or "q-nan-ni" in allowed_ids) and verify_reference_video_id("D0261_close")["eligible"]:
        alternate = reference_path("D0261_close")
        if not alternate.is_file() and (archive := index_path()) is not None:
            alternate = archive.with_name(alternate.name)
        if alternate.is_file() and (ref := _reference_variants(alternate)) is not None and ref[0].shape[2] == 2:
            alternate_scores = _variant_distances(sample, adapt(ref)).min(axis=1)
            scores["q-nan-ni"] = np.minimum(scores.get("q-nan-ni", np.inf), alternate_scores)

    ranked = sorted((float(distances.min()), entry_id) for entry_id, distances in scores.items())
    shape_trace = None
    if trace is not None:
        shape_trace = {
            "paired_frames": int(paired.sum()),
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
    max_shape_dist = 0.28 if is_rare_tier else MAX_HAND_SHAPE_DISTANCE
    # When fewer than 15/24 frames have both hands visible, single-hand frames shrink 5x in
    # _relative_hand_features, so require a tight match (<=0.22) unless palms are clasped upright.
    if paired_n < 15 and not clasped_upright:
        max_shape_dist = min(max_shape_dist, 0.22)
    if not ranked or ranked[0][0] > max_shape_dist:
        if shape_trace is not None:
            shape_trace["verdict"] = "out_of_range"
        return None
    distance, entry_id = ranked[0]
    if clasped_upright and paired_n < 12 and entry_id != "q-nan-ni":
        if shape_trace is not None:
            shape_trace["verdict"] = "clasped_low_paired_non_nanni"
        return None
    separation = MIN_HAND_SHAPE_SEPARATION
    if distance <= 0.30 and paired.mean() >= 0.80 and sample_duplicates <= 0.05:
        separation = 0.035
    elif entry_id == "q-nan-ni" and (clasped_upright or (sample_tgap is not None and sample_tgap <= 0.16 and sample_up >= 0.25)):
        separation = 0.018
    if shape_trace is not None:
        shape_trace["separation_required"] = separation
    if len(ranked) > 1 and ranked[1][0] - distance < separation:
        if shape_trace is not None:
            shape_trace["verdict"] = "ambiguous"
        return None
    # Two different endpoint trims can corroborate the same motion in one window.
    # This lets the camera use a stable match even if the next request is slow.
    first = scores[entry_id]
    rivals = [value for key, value in scores.items() if key != entry_id]
    runner_up = np.min(rivals, axis=0) if rivals else np.full_like(first, np.inf)
    agreed = (first <= max_shape_dist) & (runner_up - first >= separation)
    if is_rare_tier and agreed.sum() < 2:
        if shape_trace is not None:
            shape_trace["verdict"] = "rare_shape_uncorroborated"
            shape_trace["agreed_trims"] = int(agreed.sum())
        return None
    method = "hand_shape" if agreed.sum() >= 2 else "trajectory"
    if shape_trace is not None:
        shape_trace["verdict"] = "matched"
        shape_trace["agreed_trims"] = int(agreed.sum())
        shape_trace["match_method"] = method
    return _match_result(distance, entry_id, method)


def _match_kha_nang_with_occlusion(frames, aspect_ratio):
    """Recover 'khả năng' (D0096) when the vertical edge-on hand drops out behind the horizontal flat hand.

    In 'khả năng', one flat open hand sits horizontally across the chest while the other flat hand
    stands vertically perpendicular ('⊥') behind/against it. Turning the vertical hand edge-on to
    a webcam frequently hides its 21 hand landmarks while Holistic pose wrists/indices (pose[15..20])
    remain tracked continuously.
    """
    try:
        clip, _ = _visible_clip(frames)
    except ValueError:
        return None
    clip = _fill_short_gaps(clip)
    clip[..., 1] /= aspect_ratio
    hands = _tracked_hands(clip)
    body = normalize_sequence_tjc_v2(clip)
    if len(body) < 8:
        return None

    best_score = None
    for horiz_side in (0, 1):
        vert_side = 1 - horiz_side
        h_start = 33 if horiz_side == 0 else 54
        v_start = 54 if horiz_side == 0 else 33
        h_pw_idx = 15 if horiz_side == 0 else 16
        v_pw_idx = 16 if horiz_side == 0 else 15
        v_pi_idx = 20 if horiz_side == 0 else 19

        h_ok = hands[:, horiz_side]
        if h_ok.sum() < 6 or h_ok.mean() < 0.45:
            continue

        h_hand = body[:, h_start:h_start + 21, :2]
        h_wrist = h_hand[:, 0]
        h_palm = np.maximum(np.linalg.norm(h_hand[:, 9] - h_wrist, axis=1), 1e-4)
        h_tips = h_hand[:, [8, 12, 16, 20]] - h_wrist[:, None, :]
        h_ext = np.linalg.norm(h_tips, axis=2).mean(axis=1) / h_palm
        h_vec = h_hand[:, 12] - h_wrist
        h_len = np.maximum(np.linalg.norm(h_vec, axis=1), 1e-4)
        h_horiz = np.abs(h_vec[:, 0]) / h_len
        h_spread = np.linalg.norm(h_hand[:, 8] - h_hand[:, 20], axis=1) / h_palm
        horiz_frame = h_ok & (h_wrist[:, 1] <= 0.72) & (h_ext >= 1.35) & (h_horiz >= 0.78) & (h_spread <= 1.25)
        if horiz_frame.sum() < 5:
            continue

        v_ok = hands[:, vert_side]
        v_pw = body[:, v_pw_idx, :2]
        v_pi = body[:, v_pi_idx, :2]
        v_pose_ok = np.any(v_pw != 0, axis=1) & np.any(v_pi != 0, axis=1) & (v_pw[:, 1] <= 0.75)

        v_pos = np.where(v_ok[:, None], body[:, v_start, :2], v_pw)
        v_dir_hand = body[:, v_start + 12, :2] - body[:, v_start, :2]
        v_dir_pose = v_pi - v_pw
        v_dir = np.where(v_ok[:, None], v_dir_hand, v_dir_pose)
        v_present = v_ok | v_pose_ok

        v_len = np.maximum(np.linalg.norm(v_dir, axis=1), 1e-4)
        v_up = (-v_dir[:, 1]) / v_len
        cos_theta = np.abs((h_vec[:, 0] * v_dir[:, 0] + h_vec[:, 1] * v_dir[:, 1]) / (h_len * v_len))

        h_mid = 0.5 * (h_wrist + h_hand[:, 9])
        h_pw = body[:, h_pw_idx, :2]
        dist_hand = np.minimum(
            np.linalg.norm(v_pos - h_wrist, axis=1),
            np.linalg.norm(v_pos - h_mid, axis=1),
        )
        dist_pose = np.where(
            np.any(h_pw != 0, axis=1) & np.any(v_pw != 0, axis=1),
            np.linalg.norm(v_pw - h_pw, axis=1),
            np.inf,
        )
        pair_dist = np.minimum(dist_hand, dist_pose)
        perp_frame = horiz_frame & v_present & (pair_dist <= 0.55) & (v_up >= 0.42) & (cos_theta <= 0.75)
        perp_n = int(perp_frame.sum())
        if perp_n < 6 or perp_frame.mean() < 0.28:
            continue

        # Vertical hand must have been tracked pointing upward at least twice so 1-hand signs never match.
        v_hand_len = np.maximum(np.linalg.norm(v_dir_hand, axis=1), 1e-4)
        v_hand_up_n = int((v_ok & ((-v_dir_hand[:, 1]) / v_hand_len >= 0.45)).sum())
        if v_hand_up_n < 2:
            continue

        dist_med = float(np.median(pair_dist[perp_frame]))
        h_spread_med = float(np.median(h_spread[horiz_frame]))
        h_ext_med = float(np.median(h_ext[horiz_frame]))
        if dist_med > 0.35 or h_spread_med > 0.95 or h_ext_med < 1.50:
            continue

        # Require real gesture motion (hands converging into contact or sliding horizontally during contact).
        valid_dists = pair_dist[np.isfinite(pair_dist)]
        converged = float(valid_dists.max() - valid_dists.min()) >= 0.12 if len(valid_dists) >= 4 else False
        slid = float(np.ptp(h_wrist[perp_frame, 0])) >= 0.12
        if not (converged or slid):
            continue

        cos_med = float(np.median(cos_theta[perp_frame]))
        distance = min(0.32, 0.14 + 0.18 * dist_med + 0.12 * cos_med)
        if best_score is None or distance < best_score:
            best_score = distance

    if best_score is None:
        return None
    return _match_result(best_score, "q-kha-nang", method="hand_shape")


def _match_chung_toi_family(frames, aspect_ratio, allowed_set=None):
    """Recognize 'chúng tôi' and 'chúng tôi (2/3/4/5 người)' across studio and close-up webcam framing.

    In QIPEDC studio videos (D0130-D0134), the signer performs a chest-level horizontal circular sweep
    with 1, 2, 3, 4, or 5 extended fingers while holding the arm outside the shoulder (which falls outside
    a 4:3 webcam frame, so webcam users naturally circle in front of the chest). Requiring a closed/bidirectional
    horizontal loop rejects one-way crawls (e.g. 'con sâu') and vertical taps (e.g. 'ăn đủ', 'bực mình').
    """
    if allowed_set is not None and not any(eid in allowed_set for eid in _CHUNG_TOI_IDS):
        return None
    try:
        clip, _ = _visible_clip(frames)
    except ValueError:
        return None
    clip = _fill_short_gaps(clip)
    clip[..., 1] /= aspect_ratio
    obs = _tracked_hands(clip)
    body = normalize_sequence_tjc_v2(clip)
    if len(body) < 10:
        return None

    mov = np.zeros(2)
    ymed = np.full(2, 9.9)
    for side in (0, 1):
        pts = body[obs[:, side], 33 + 21 * side:33 + 21 * (side + 1), :2]
        if len(pts) >= 5:
            low, high = np.percentile(pts, [5, 95], axis=0)
            mov[side] = float(np.linalg.norm(high - low, axis=-1).mean())
            ymed[side] = float(np.median(pts[:, 0, 1]))
    s = int(np.argmax(mov))
    other = 1 - s
    if mov[s] < 0.10:
        return None
    if obs[:, other].sum() >= 6 and ymed[other] <= 0.65 and mov[other] >= max(0.08, 0.28 * mov[s]):
        return None
    other_pw = 16 if s == 0 else 15
    pw_ok = np.any(body[:, other_pw, :2] != 0, axis=1)
    if float((pw_ok & (body[:, other_pw, 1] <= 0.65)).mean()) > 0.25:
        return None

    j0 = 33 + 21 * s
    ok_idx = np.flatnonzero(obs[:, s] & (body[:, j0, 1] <= RAISED_WRIST_MAX_Y))
    if len(ok_idx) < 8:
        return None
    h_all = body[ok_idx, j0:j0 + 21, :2].copy()
    if s == 0:
        h_all[..., 0] *= -1
    trim = len(ok_idx) // 6 if len(ok_idx) >= 14 else 0
    h = h_all[trim:len(h_all) - trim] if trim else h_all

    w = h[:, 0]
    psize = np.maximum(np.linalg.norm(h[:, 9] - w, axis=1), 1e-4)
    d_vals = {}
    st_vals = {}
    mixed_transition = False
    for fn, ti, pi, mi in (("I", 8, 6, 5), ("M", 12, 10, 9), ("R", 16, 14, 13), ("P", 20, 18, 17)):
        dt = np.linalg.norm(h[:, ti] - w, axis=1) / psize
        dp = np.linalg.norm(h[:, pi] - w, axis=1) / psize
        diff_tp = dt - dp
        if fn == "M":
            p15, p75 = np.percentile(diff_tp, [15, 75])
            mixed_transition = bool(p15 < -0.05 and p75 > 0.22)
        seg = np.linalg.norm(h[:, ti] - h[:, pi], axis=1) + np.linalg.norm(h[:, pi] - h[:, mi], axis=1)
        st = np.linalg.norm(h[:, ti] - h[:, mi], axis=1) / np.maximum(seg, 1e-4)
        d_vals[fn] = float(np.median(diff_tp))
        st_vals[fn] = float(np.median(st))
    if mixed_transition:
        return None

    mrp_curled = all(d_vals[k] <= -0.08 and st_vals[k] <= 0.72 for k in ("M", "R", "P"))
    is_one_finger = mrp_curled and (
        st_vals["I"] >= 0.65 or d_vals["I"] >= max(d_vals["M"], d_vals["R"], d_vals["P"]) + 0.04
    )

    wrist_y = float(np.median(h[:, 0, 1]))
    tips_y = float(np.median(h[:, [8, 12, 16, 20], 1].mean(axis=1)))
    if is_one_finger:
        if wrist_y < -0.04 or wrist_y > 0.65 or tips_y < -0.02:
            return None
    elif len(ok_idx) < 10 or wrist_y < 0.01 or wrist_y > 0.62 or tips_y < 0.04:
        return None

    palm_c = h_all[:, 9]
    idx_tip = h_all[:, 8]
    dx_p = np.diff(palm_c[:, 0])
    dx_i = np.diff(idx_tip[:, 0])
    pos_p, neg_p = float(dx_p[dx_p > 0].sum()), float((-dx_p[dx_p < 0]).sum())
    pos_i, neg_i = float(dx_i[dx_i > 0].sum()), float((-dx_i[dx_i < 0]).sum())
    xp_p = float(np.ptp(palm_c[:, 0]))
    xp_i = float(np.ptp(idx_tip[:, 0]))
    path_h = float(np.linalg.norm(np.diff(h[:, 8], axis=0), axis=1).sum())
    net_h = float(np.linalg.norm(h[-1, 8] - h[0, 8]))
    if is_one_finger:
        has_loop = (
            xp_i >= 0.24
            and xp_p >= 0.14
            and min(pos_i, neg_i) >= 0.36
            and min(pos_p, neg_p) >= 0.20
            and path_h / max(net_h, 0.05) >= 1.5
        )
    else:
        has_loop = (
            xp_i >= 0.38
            and xp_p >= 0.22
            and min(pos_i, neg_i) >= 0.72
            and min(pos_p, neg_p) >= 0.42
            and path_h / max(net_h, 0.05) >= 1.8
        )
    if not has_loop:
        return None

    entry_id = None
    if is_one_finger:
        entry_id = "q-chung-toi-dai-tu"
    else:
        max_d = max(d_vals.values())
        if max_d < 0.25:
            return None
        t_p20 = float(np.median(np.linalg.norm(h[:, 4] - h[:, 20], axis=1) / psize))
        t_i6 = float(np.median(np.linalg.norm(h[:, 4] - h[:, 6], axis=1) / psize))
        lat = h[:, 5] - h[:, 17]
        lat_unit = lat / np.maximum(np.linalg.norm(lat, axis=1, keepdims=True), 1e-4)
        thumb_out = float(np.median(np.sum((h[:, 4] - h[:, 5]) * lat_unit, axis=1) / psize))

        ext_4 = [fn for fn in ("I", "M", "R", "P") if d_vals[fn] >= 0.18 and st_vals[fn] >= 0.82]
        if "P" in ext_4 and t_p20 <= 1.05 and d_vals["P"] <= 0.55 * max_d:
            ext_4.remove("P")
        if len(ext_4) == 2 and "I" in ext_4 and "M" in ext_4:
            entry_id = "q-chung-toi-3-nguoi-dai-tu" if (thumb_out >= 0.28 and t_i6 >= 0.75) else "q-chung-toi-2-nguoi-dai-tu"
        elif len(ext_4) == 3:
            entry_id = "q-chung-toi-3-nguoi-dai-tu"
        elif len(ext_4) == 4:
            entry_id = "q-chung-toi-5-nguoi-dai-tu" if (thumb_out >= 0.22 and t_p20 >= 1.80) else "q-chung-toi-4-nguoi-dai-tu"

    if entry_id is None or (allowed_set is not None and entry_id not in allowed_set):
        return None
    return _match_result(0.27, entry_id, method="hand_shape")


def _match_head_salute(frames, aspect_ratio, allowed_set=None, *, allow_rare=True):
    """Distinguish forehead-outward salute ('chào' / 'xin chào') from above-head-into-forehead ('tuân lệnh')."""
    allow_chao = allowed_set is None or "q-chao" in allowed_set
    allow_tuan_lenh = allow_rare and (allowed_set is None or "q-tuan-lenh" in allowed_set)
    if not (allow_chao or allow_tuan_lenh):
        return None
    try:
        clip, _ = _visible_clip(frames)
    except ValueError:
        return None
    clip = _fill_short_gaps(clip)
    clip[..., 1] /= aspect_ratio
    obs = _tracked_hands(clip)
    body = normalize_sequence_tjc_v2(clip)
    if len(body) < 8:
        return None

    mov = np.zeros(2)
    ymed = np.full(2, 9.9)
    for side in (0, 1):
        pts = body[obs[:, side], 33 + 21 * side:33 + 21 * (side + 1), :2]
        if len(pts) >= 4:
            low, high = np.percentile(pts, [5, 95], axis=0)
            mov[side] = float(np.linalg.norm(high - low, axis=-1).mean())
            ymed[side] = float(np.median(pts[:, 0, 1]))
    s = int(np.argmax(mov))
    other = 1 - s
    if mov[s] < 0.08:
        return None
    if obs[:, other].sum() >= 6 and ymed[other] <= 0.65 and mov[other] >= max(0.08, 0.28 * mov[s]):
        return None
    other_pw = 16 if s == 0 else 15
    pw_ok = np.any(body[:, other_pw, :2] != 0, axis=1)
    if float((pw_ok & (body[:, other_pw, 1] <= 0.65)).mean()) > 0.25:
        return None

    eye_idx = 2 if s == 0 else 5
    eye_ok = np.any(body[:, eye_idx, :2] != 0, axis=1)
    if eye_ok.mean() < 0.70:
        return None

    ok = np.flatnonzero(obs[:, s] & eye_ok)
    if len(ok) < 7:
        return None
    h = body[ok, 33 + 21 * s:33 + 21 * (s + 1), :2].copy()
    eye = body[ok, eye_idx, :2].copy()
    if s == 0:
        h[..., 0] *= -1
        eye[:, 0] *= -1

    fh = eye + np.array([-0.04, -0.10], dtype=np.float32)
    tip = 0.5 * (h[:, 8] + h[:, 12])
    wrist = h[:, 0]
    if float(np.max(tip[:, 0])) > 0.22 or float(np.min(wrist[:, 1])) > 0.06:
        return None

    upper = tip[:, 1] <= (eye[:, 1] + 0.16)
    if int(upper.sum()) < 5:
        return None
    h_u, tip_u, fh_u, eye_u = h[upper], tip[upper], fh[upper], eye[upper]
    d_fh = np.linalg.norm(tip_u - fh_u, axis=1)
    if float(d_fh.min()) > 0.24:
        return None

    # Flat adducted salute blade (I, M, R straight & extended, I-M held close together unlike open-5 shake)
    w_u = h_u[:, 0]
    psize = np.maximum(np.linalg.norm(h_u[:, 9] - w_u, axis=1), 1e-4)
    im_gap = float(np.median(np.linalg.norm(h_u[:, 8] - h_u[:, 12], axis=1) / psize))
    if im_gap > 0.46:
        return None
    for ti, pi, mi in ((8, 6, 5), (12, 10, 9), (16, 14, 13)):
        dt = np.linalg.norm(h_u[:, ti] - w_u, axis=1) / psize
        dp = np.linalg.norm(h_u[:, pi] - w_u, axis=1) / psize
        seg = np.linalg.norm(h_u[:, ti] - h_u[:, pi], axis=1) + np.linalg.norm(h_u[:, pi] - h_u[:, mi], axis=1)
        st = np.linalg.norm(h_u[:, ti] - h_u[:, mi], axis=1) / np.maximum(seg, 1e-4)
        if float(np.median(dt - dp)) < 0.12 or float(np.median(st)) < 0.80:
            return None

    high = tip_u[:, 1] <= (eye_u[:, 1] + 0.06)
    if int(high.sum()) < 4:
        return None
    d_hi, tip_hi, fh_hi, ps_hi = d_fh[high], tip_u[high], fh_u[high], psize[high]

    contact_thr = min(0.24, float(d_hi.min()) + 0.06)
    contacts = np.flatnonzero(d_hi <= contact_thr)
    if not len(contacts):
        return None
    c_first, c_last = int(contacts[0]), int(contacts[-1])
    c_dist = float(d_hi[contacts].min())
    c_palm = float(np.median(ps_hi[contacts]))

    after = d_hi[c_first + 1:]
    after_ps = ps_hi[c_first + 1:]
    away_frames = int((after >= c_dist + 0.11).sum())
    zoom_frames = int((after_ps >= c_palm * 1.18).sum())
    tail_d = float(np.median(d_hi[max(0, len(d_hi) - 3):]))

    before_u = d_fh[:int(np.flatnonzero(high)[c_last]) + 1]
    inward_2d = (float(before_u.max()) - c_dist) if len(before_u) >= 2 else 0.0
    started_above_fh = bool(float(np.min(tip_hi[:c_last + 1, 1] - fh_hi[:c_last + 1, 1])) <= -0.05)

    entry_id = None
    if (away_frames >= 2 or zoom_frames >= 2) and tail_d >= c_dist + 0.09:
        if allow_chao:
            entry_id = "q-chao"
    elif away_frames <= 1 and zoom_frames <= 1 and tail_d <= 0.22 and (
        started_above_fh or inward_2d >= 0.10 or len(contacts) >= 4
    ):
        if allow_tuan_lenh:
            entry_id = "q-tuan-lenh"

    if entry_id is None:
        return None
    return _match_result(0.24, entry_id, method="hand_shape")


def corroborate_sstcn_entry(frames, aspect_ratio, entry_id):
    """Reject out-of-distribution SSTCN guesses (e.g. 'Em gái' on 'chúng tôi', 'Xin chào' on 'tuân lệnh')
    when the predicted entry exists in the QIPEDC live index and the actual motion disagrees."""
    if not entry_id:
        return True
    idx = _active_index()
    if idx is None or entry_id not in idx["id_set"]:
        return True
    if entry_id == "q-chao":
        salute = _match_head_salute(frames, aspect_ratio)
        if salute is not None and salute.get("entry_id") == "q-tuan-lenh":
            return False
    if entry_id == "q-em-gai":
        try:
            clip, _ = _visible_clip(frames)
            clip = _fill_short_gaps(clip)
            clip[..., 1] /= aspect_ratio
            obs = _tracked_hands(clip)
            body = normalize_sequence_tjc_v2(clip)
            min_tip_y = np.inf
            for side in (0, 1):
                if obs[:, side].any():
                    j0 = 33 + 21 * side
                    tips_y = body[obs[:, side], j0 + 8:j0 + 21:4, 1].mean(axis=1)
                    min_tip_y = min(min_tip_y, float(tips_y.min()))
            # Genuine 'em gái' raises fingers to ear/cheek level (y <= -0.05); chest-only signs stay > +0.02.
            if min_tip_y > 0.02:
                return False
        except ValueError:
            return False

    recording = _variants(frames, aspect_ratio)
    if recording is None:
        return False
    hands = recording[0].shape[2]
    bucket = idx["buckets"].get(hands)
    if bucket is None or entry_id not in bucket["ids"]:
        return False
    pos = bucket["ids"].index(entry_id)
    st, en = bucket["slices"][pos]
    ref = (
        bucket["features"][st:en].astype(np.float32, copy=False),
        bucket["observed"][st:en],
        bucket["penalties"][st:en],
    )
    min_dist = float(_variant_distances(recording, ref).min())
    return min_dist <= (0.50 if hands == 1 else 0.60)


def _effective_unilateral_separation(best_distance):
    """Scale separation for very tight 1-hand matches (e.g. dist 0.12-0.18 in crowded chest signs)."""
    if best_distance <= 0.18:
        return 0.016
    if best_distance <= 0.22:
        return 0.028
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
            references[entry_id] = reference


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
    allow_rare_salute=True,
    trace=None,
):
    hands = recording[0].shape[2]
    bilateral = hands == 2
    references = dict(_indexed_references(recording, bucket, allowed_ids=allowed_set)) if bucket is not None else {}
    _collect_extra_references(entries, references, allowed_set, idx, include_indexed=False)
    # Ensure user-pinned entries that exist in the main index are also evaluated in Tier 1.
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
        min_separation = MIN_BILATERAL_SEPARATION
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
    salute_override = (
        _match_head_salute(frames, aspect_ratio, allowed_set, allow_rare=allow_rare_salute)
        if (not bilateral and not is_rare_tier and idx is not None)
        else None
    )
    if in_range and (len(ranked) == 1 or ranked[1][0] - ranked[0][0] >= min_separation):
        # Require >=10/24 paired resampled frames AND >=10 raw frames of the weaker hand (scaled for
        # short clips) for loose (>0.35) or rare bilateral trajectory matches, preventing brief 5-8 frame
        # overlaps (e.g. wrist-tap of "mấy giờ?", transition into "kỹ năng") from false-matching.
        paired_n = int(recording[1][0].all(axis=1).sum()) if bilateral else 24
        weak_overlap = False
        if bilateral and (is_rare_tier or ranked[0][0] > 0.35):
            raw_obs = _tracked_hands(np.asarray(frames, dtype=np.float32)).sum(axis=0)
            weak_overlap = paired_n < 10 or int(raw_obs.min()) < 10
        elif not bilateral and is_rare_tier and ranked[0][0] > 0.22:
            raw_obs = _tracked_hands(np.asarray(frames, dtype=np.float32)).sum(axis=0)
            weak_overlap = int(raw_obs.max()) < 12
        if weak_overlap:
            tier_info["verdict"] = "bilateral_too_few_paired_frames" if bilateral else "unilateral_rare_too_short"
        else:
            if ranked[0][1] in ("q-chao", "q-tuan-lenh") and salute_override is not None:
                tier_info["verdict"] = "head_salute_directional_matched"
                if trace is not None:
                    trace.update(tier_info)
                return salute_override, tier_info
            tier_info["verdict"] = "trajectory_matched"
            if trace is not None:
                trace.update(tier_info)
            return _match_result(*ranked[0]), tier_info
    if salute_override is not None:
        tier_info["verdict"] = "head_salute_matched"
        if trace is not None:
            trace.update(tier_info)
        return salute_override, tier_info
    if not bilateral and not is_rare_tier:
        if (chung_toi := _match_chung_toi_family(frames, aspect_ratio, allowed_set)) is not None:
            tier_info["verdict"] = "chung_toi_family_matched"
            if trace is not None:
                trace.update(tier_info)
            return chung_toi, tier_info
    if bilateral and not is_rare_tier and (allowed_set is None or "q-kha-nang" in allowed_set):
        if (kha_nang := _match_kha_nang_with_occlusion(frames, aspect_ratio)) is not None:
            tier_info["verdict"] = "kha_nang_occlusion_matched"
            if trace is not None:
                trace.update(tier_info)
            return kha_nang, tier_info
    # On rare 2-hand tier, skip bilateral_shape unless trajectory is already reasonably close (<=0.45),
    # preventing weak transition clips (e.g. dist=0.515 "bữa cơm") from false-matching rare templates.
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


def match_live_words(frames, aspect_ratio, entries, *, allowed_ids=None, vocab_scope="all", trace=None):
    """Accept a reference only when the absolute match and lead over rivals are sufficient.

    Uses a 2-tier search whenever the eligible pool is large (full index or regional word_scope):
    - Tier 1 searches common + builtin (+ user-pinned) words within allowed_ids (~4x faster
      and prevents rare words from hijacking or causing false ambiguity against common words).
    - Tier 2 (when vocab_scope == "all") searches the full allowed index so less-common signs
      (e.g. "bực mình") remain recognized out of the box.
    """
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
    full_bucket = idx["buckets"][hands] if idx is not None else None
    priority_bucket = full_bucket.get("priority_sub") if full_bucket is not None else None

    allow_rare_salute = vocab_scope != "common"
    # Single-tier path only when no priority sub-bucket exists or allowed_set is already a tiny subset (<=40).
    if priority_bucket is None or (allowed_set is not None and len(allowed_set) <= 40):
        res, info = _evaluate_tier(
            frames, aspect_ratio, recording, full_bucket, entries, idx,
            allowed_set=allowed_set, is_rare_tier=False, allow_rare_salute=allow_rare_salute, trace=trace,
        )
        if trace is not None and (res is None or res.get("status") == "ambiguous_word"):
            trace.update(info)
        return res

    # Tier 1: Fast priority (common + builtin + user-pinned entries, respecting word_scope allowed_set)
    res_t1, info_t1 = _evaluate_tier(
        frames, aspect_ratio, recording, priority_bucket, entries, idx,
        allowed_set=allowed_set, is_rare_tier=False, allow_rare_salute=allow_rare_salute, trace=trace,
    )
    if res_t1 and res_t1.get("status") == "word_matched":
        return res_t1

    if vocab_scope == "common":
        if trace is not None:
            trace.update(info_t1)
        return res_t1

    # Tier 2: Fallback to full eligible index for less-common signs (e.g. "bực mình")
    res_t2, info_t2 = _evaluate_tier(
        frames, aspect_ratio, recording, full_bucket, entries, idx,
        allowed_set=allowed_set, is_rare_tier=True, trace=trace,
    )
    if res_t2 and res_t2.get("status") == "word_matched":
        return res_t2

    # Prefer whichever tier had an in-range candidate for ambiguity/suggestions reporting.
    chosen_res = res_t2 if res_t2 is not None else res_t1
    chosen_info = info_t2 if res_t2 is not None else (info_t1 if res_t1 is not None else info_t2)
    if trace is not None:
        trace.update(chosen_info)
        trace["tier1_top1"] = (info_t1.get("top_candidates") or [None])[0]
    return chosen_res
