"""Compare webcam motion with a library of prepared QIPEDC references."""

from functools import lru_cache
import os
from pathlib import Path

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


DEFAULT_WORDS = (
    "q-an-du", "q-an-vua", "q-an-it", "q-com-2", "q-nem", "q-lau-mieng", "q-thom",
    "q-nan-ni", "q-cau-nguyen", "q-vai-lay",
    "q-xin-loi", "q-ky-nang", "q-kha-nang", "q-khong-co",
)
BUILTIN_WORDS = (TIME_SIGN_ENTRY, *DEFAULT_WORDS)
MAX_EXTRA_WORDS = 12
FRAMES = 24
COARSE_BINS = 6
STAGE1_TOP_K = 24
DTW_REFERENCE_BATCH_SIZE = 6
MAX_DISTANCE = 0.35
MIN_SEPARATION = 0.05
MAX_BILATERAL_DISTANCE = 0.52
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


def _variants(raw, aspect_ratio, *, reference=False):
    try:
        clip, _ = _visible_clip(raw)
    except ValueError:
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
    for side in (0, 1):
        points = hands[observed[:, side], side, :, :2]
        if len(points) >= 5:
            low, high = np.percentile(points, [5, 95] if not reference else [10, 90], axis=0)
            movement[side] = np.linalg.norm(high - low, axis=-1).mean()

    active = (movement >= 0.09) & (movement >= 0.23 * max(movement.max(), 1e-6))
    if not active.all() and (observed.sum(axis=0) >= 2).all():
        # Retain bilateral structure when two hands interact and one is mostly occluded (e.g. "năn nỉ").
        active[:] = True
    observed = observed[:, active]
    tracked = np.flatnonzero(observed.any(axis=1))
    if len(tracked) < 7 or observed[tracked[0]:tracked[-1] + 1].any(axis=1).mean() < 0.70:
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
    if not reference:
        active_trim = _active_motion_trim(features, observed)
        if active_trim is not None and active_trim not in trims:
            trims.append(active_trim)
            penalty_list.append(0.015)
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


@lru_cache(maxsize=2)
def _load_index_cached(path_str, mtime_ns):
    del mtime_ns
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
            for key in ("coarse", "relative_coarse"):
                if key in bucket:
                    bucket[key + "_squared"] = np.einsum("vthd,vthd->vth", bucket[key], bucket[key])
    return {"buckets": buckets, "all_ids": tuple(all_ids), "id_set": frozenset(all_ids)}


def _active_index():
    path = index_path()
    if path is None:
        return None
    try:
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
    sample_f, sample_o, sample_cut = recording
    sample_coarse = _coarse_pool(sample_f, sample_o)  # [V_s, 6, H, 14]
    tmpl_coarse = bucket[coarse_key]                    # [N_v, 6, H, 14]
    tmpl_squared = bucket[coarse_key + "_squared"]
    tmpl_pen = bucket["penalties"]                    # [N_v]
    best_var = np.full(len(tmpl_coarse), np.inf, dtype=np.float32)
    for sv, cut in zip(sample_coarse, sample_cut):
        dot = np.einsum("vthd,thd->vth", tmpl_coarse, sv)
        squared = tmpl_squared + (sv * sv).sum(axis=-1) - 2 * dot
        diff = np.sqrt(np.maximum(squared, 0)).mean(axis=(1, 2)) + tmpl_pen + float(cut)
        np.minimum(best_var, diff, out=best_var)
    entry_best = np.full(n_entries, np.inf, dtype=np.float32)
    np.minimum.at(entry_best, bucket["var_to_entry"], best_var)
    if allowed_ids is not None:
        allowed_idx = [i for i, eid in enumerate(bucket["ids"]) if eid in allowed_ids]
        if not allowed_idx:
            return []
        if len(allowed_idx) <= top_k:
            return allowed_idx
        sub = entry_best[allowed_idx]
        chosen = np.argpartition(sub, top_k)[:top_k]
        return [allowed_idx[int(k)] for k in chosen]
    if n_entries <= top_k:
        return list(range(n_entries))
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


def _bilateral_shape_match(frames, aspect_ratio, recording, bucket, references, *, allowed_ids=None):
    f, o, p = recording
    paired = o[0].all(axis=1)
    if paired.sum() < 5:
        return None
    sample = (_relative_hand_features(f, o), o, p)
    sample_duplicates = _duplicate_hand_fraction(f, o)

    def adapt(reference):
        feat, observed, penalties = reference
        duplicates = max(0, _duplicate_hand_fraction(feat, observed) - sample_duplicates)
        return _relative_hand_features(feat, observed), observed, penalties + 0.20 * duplicates

    candidates = dict(references)
    if bucket is not None:
        candidates.update(_indexed_references(sample, bucket, coarse_key="relative_coarse", allowed_ids=allowed_ids))
    scores = _score_templates(sample, {key: adapt(ref) for key, ref in candidates.items() if ref[0].shape[2] == 2})

    # A closer view of the public năn nỉ sample preserves fingertips during contact.
    if allowed_ids is None or "q-nan-ni" in allowed_ids:
        alternate = reference_path("D0261_close")
        if not alternate.is_file() and (archive := index_path()) is not None:
            alternate = archive.with_name(alternate.name)
        if alternate.is_file() and (ref := _reference_variants(alternate)) is not None and ref[0].shape[2] == 2:
            alternate_scores = _variant_distances(sample, adapt(ref)).min(axis=1)
            scores["q-nan-ni"] = np.minimum(scores.get("q-nan-ni", np.inf), alternate_scores)

    ranked = sorted((float(distances.min()), entry_id) for entry_id, distances in scores.items())
    if not ranked or ranked[0][0] > MAX_HAND_SHAPE_DISTANCE:
        return None
    distance, entry_id = ranked[0]
    separation = MIN_HAND_SHAPE_SEPARATION
    if distance <= 0.30 and paired.mean() >= 0.80 and sample_duplicates <= 0.05:
        separation = 0.020
    elif entry_id == "q-nan-ni" and lost_hand_during_interaction(frames, aspect_ratio):
        separation = 0.018
    if len(ranked) > 1 and ranked[1][0] - distance < separation:
        return None
    # Two different endpoint trims can corroborate the same motion in one window.
    # This lets the camera use a stable match even if the next request is slow.
    first = scores[entry_id]
    rivals = [value for key, value in scores.items() if key != entry_id]
    runner_up = np.min(rivals, axis=0) if rivals else np.full_like(first, np.inf)
    agreed = (first <= MAX_HAND_SHAPE_DISTANCE) & (runner_up - first >= separation)
    method = "hand_shape" if agreed.sum() >= 2 else "trajectory"
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


def match_live_words(frames, aspect_ratio, entries, *, allowed_ids=None):
    """Accept a reference only when the absolute match and lead over rivals are sufficient."""
    if (recording := _variants(frames, aspect_ratio)) is None:
        return None
    allowed_set = frozenset(allowed_ids) if allowed_ids is not None else None
    hands = recording[0].shape[2]
    idx = _active_index()
    bucket = idx["buckets"][hands] if idx is not None else None
    references = dict(_indexed_references(recording, bucket, allowed_ids=allowed_set)) if bucket is not None else {}

    for entry_id, video_id in entries or ():
        if allowed_set is not None and entry_id not in allowed_set:
            continue
        if entry_id in references:
            continue
        if idx is not None and entry_id in idx["id_set"]:
            continue
        path = reference_path(video_id)
        if not path.is_file():
            continue
        reference = _reference_variants(path)
        if reference is not None:
            references[entry_id] = reference

    ranked = sorted((float(distances.min()), entry_id) for entry_id, distances in _score_templates(recording, references).items())
    bilateral = hands == 2
    max_distance = MAX_BILATERAL_DISTANCE if bilateral else MAX_DISTANCE
    min_separation = MIN_BILATERAL_SEPARATION if bilateral else MIN_SEPARATION
    in_range = bool(ranked and ranked[0][0] <= max_distance)
    if in_range and (len(ranked) == 1 or ranked[1][0] - ranked[0][0] >= min_separation):
        return _match_result(*ranked[0])
    if bilateral and (allowed_set is None or "q-kha-nang" in allowed_set):
        if (kha_nang := _match_kha_nang_with_occlusion(frames, aspect_ratio)) is not None:
            return kha_nang
    if bilateral and (recovered := _bilateral_shape_match(frames, aspect_ratio, recording, bucket, references, allowed_ids=allowed_set)):
        return recovered
    if in_range:
        return {"motion_detected": True, "status": "ambiguous_word"}
    return None
