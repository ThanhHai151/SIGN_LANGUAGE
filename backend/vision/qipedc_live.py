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
    "q-xin-loi", "q-ky-nang", "q-kha-nang",
)
BUILTIN_WORDS = (TIME_SIGN_ENTRY, *DEFAULT_WORDS)
MAX_EXTRA_WORDS = 12
FRAMES = 24
COARSE_BINS = 6
STAGE1_TOP_K = 24
MAX_DISTANCE = 0.35
MIN_SEPARATION = 0.05
MAX_BILATERAL_DISTANCE = 0.52
MIN_BILATERAL_SEPARATION = 0.08
HAND_MISMATCH_PENALTY = 0.10
RAISED_WRIST_MAX_Y = 0.68
REFLECT_X = np.tile([-1, 1], 7).astype(np.float32)

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
            low, high = np.percentile(points, [10, 90], axis=0)
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
        0.60 * (wrist - mouth[:, None]),
        0.95 * (tips.mean(axis=2) - mouth[:, None]),
        0.22 * shape.reshape(len(clip), -1, 10),
    ), axis=-1)
    if not bilateral and active[0]:
        features *= REFLECT_X

    usable = slice(tracked[0], tracked[-1] + 1) if bilateral else tracked
    features, observed = features[usable], observed[usable]
    cuts = sorted({0, len(features) // 6})
    trims = [(start, end) for start in cuts for end in cuts]
    indices = np.rint([
        np.linspace(start, len(features) - end - 1, FRAMES) for start, end in trims
    ]).astype(int)
    features, observed = features[indices], observed[indices]
    penalties = np.array([0.025 if start or end else 0.0 for start, end in trims], dtype=np.float32)
    if bilateral and reference:
        features = np.concatenate((features, features[:, :, ::-1] * REFLECT_X))
        observed = np.concatenate((observed, observed[:, :, ::-1]))
        penalties = np.tile(penalties, 2)
    return features, observed, penalties


@lru_cache(maxsize=96)
def _reference_variants(path):
    return _variants(np.load(path, allow_pickle=False), 1, reference=True)


def prepare_live_reference(video_id):
    prepare_reference(video_id)
    return video_id == TIME_SIGN_VIDEO or _reference_variants(reference_path(video_id)) is not None


def _distance(recording, reference):
    sample, sample_hands, cut = recording
    template, template_hands, ref_cut = reference
    if sample.shape[2] != template.shape[2]:
        return np.inf
    together = sample_hands[:, None, :, None] & template_hands[None, :, None, :]
    different = sample_hands[:, None, :, None] ^ template_hands[None, :, None, :]
    delta = np.linalg.norm(sample[:, None, :, None] - template[None, :, None, :], axis=-1)
    cost = (np.minimum(delta, 3) * together + HAND_MISMATCH_PENALTY * different).sum(axis=-1)
    cost /= np.maximum(1, together.sum(axis=-1))
    cost += 0.7 * ~together.any(axis=-1)
    # Score all short endpoint trims together. Restrict the temporal warp
    # so that matching a single pose cannot explain the entire trajectory.
    dp = np.full((FRAMES + 1, FRAMES + 1, len(sample), len(template)), np.inf, dtype=np.float32)
    dp[0, 0] = 0
    for i in range(1, FRAMES + 1):
        for j in range(max(1, i - 5), min(FRAMES, i + 5) + 1):
            dt = cost[:, :, i - 1, j - 1]
            dp[i, j] = np.minimum(dp[i - 1, j - 1] + dt, np.minimum(dp[i - 1, j], dp[i, j - 1])) + dt
    return float((dp[-1, -1] / (2 * FRAMES) + cut[:, None] + ref_cut).min())


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


@lru_cache(maxsize=2)
def _load_index_cached(path_str, mtime_ns):
    del mtime_ns
    with np.load(path_str, allow_pickle=False) as data:
        buckets = {}
        all_ids = []
        for hands, prefix in ((1, "uni"), (2, "bi")):
            ids = [str(x) for x in data[f"{prefix}_ids"].tolist()]
            all_ids.extend(ids)
            feats = data[f"{prefix}_features"].astype(np.float32)
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
            buckets[hands] = {
                "ids": ids, "features": feats, "observed": obs,
                "penalties": pens, "var_to_entry": v2e, "coarse": coarse, "slices": slices,
            }
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


def _stage1_candidates(recording, bucket, top_k=STAGE1_TOP_K):
    n_entries = len(bucket["ids"])
    if n_entries == 0:
        return []
    sample_f, sample_o, sample_cut = recording
    sample_coarse = _coarse_pool(sample_f, sample_o)  # [V_s, 6, H, 14]
    tmpl_coarse = bucket["coarse"]                    # [N_v, 6, H, 14]
    tmpl_pen = bucket["penalties"]                    # [N_v]
    best_var = np.full(len(tmpl_coarse), np.inf, dtype=np.float32)
    for sv, cut in zip(sample_coarse, sample_cut):
        diff = np.linalg.norm(tmpl_coarse - sv[None], axis=-1).mean(axis=(1, 2)) + tmpl_pen + float(cut)
        np.minimum(best_var, diff, out=best_var)
    entry_best = np.full(n_entries, np.inf, dtype=np.float32)
    np.minimum.at(entry_best, bucket["var_to_entry"], best_var)
    if n_entries <= top_k:
        return list(range(n_entries))
    return np.argpartition(entry_best, top_k)[:top_k].tolist()


def match_live_words(frames, aspect_ratio, entries):
    """Accept a reference only when the absolute match and lead over rivals are sufficient."""
    if (recording := _variants(frames, aspect_ratio)) is None:
        return None
    hands = recording[0].shape[2]
    ranked = []
    seen = set()

    idx = _active_index()
    if idx is not None:
        bucket = idx["buckets"].get(hands)
        if bucket and bucket["ids"]:
            for entry_idx in _stage1_candidates(recording, bucket):
                entry_id = bucket["ids"][entry_idx]
                s, e = bucket["slices"][entry_idx]
                ref = (bucket["features"][s:e], bucket["observed"][s:e], bucket["penalties"][s:e])
                ranked.append((_distance(recording, ref), entry_id))
                seen.add(entry_id)

    for entry_id, video_id in entries or ():
        if entry_id in seen:
            continue
        if idx is not None and entry_id in idx["id_set"]:
            continue
        path = reference_path(video_id)
        if not path.is_file():
            continue
        reference = _reference_variants(path)
        if reference is not None:
            ranked.append((_distance(recording, reference), entry_id))
            seen.add(entry_id)

    ranked.sort()
    bilateral = hands == 2
    max_distance = MAX_BILATERAL_DISTANCE if bilateral else MAX_DISTANCE
    min_separation = MIN_BILATERAL_SEPARATION if bilateral else MIN_SEPARATION
    if not ranked or ranked[0][0] > max_distance:
        return None
    if len(ranked) > 1 and ranked[1][0] - ranked[0][0] < min_separation:
        return {"motion_detected": True, "status": "ambiguous_word"}
    distance, entry_id = ranked[0]
    return {"motion_detected": True, "status": "word_matched", "entry_id": entry_id, "match_method": "trajectory",
            "similarity": round(1.0 - distance, 3)}
