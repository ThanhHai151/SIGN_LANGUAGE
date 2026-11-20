"""Compare webcam motion with a small library of prepared QIPEDC references."""

from functools import lru_cache

import numpy as np

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
)
BUILTIN_WORDS = (TIME_SIGN_ENTRY, *DEFAULT_WORDS)
MAX_EXTRA_WORDS = 12
FRAMES = 24
MAX_DISTANCE = 0.35
MIN_SEPARATION = 0.05
REFLECT_X = np.tile([-1, 1], 7).astype(np.float32)


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
    hands = body[:, 33:].reshape(len(body), 2, 21, 3)
    movement = np.zeros(2)
    for side in (0, 1):
        points = hands[observed[:, side], side, :, :2]
        if len(points) >= 5:
            low, high = np.percentile(points, [10, 90], axis=0)
            movement[side] = np.linalg.norm(high - low, axis=-1).mean()

    active = (movement >= 0.09) & (movement >= 0.23 * movement.max())
    observed = observed[:, active]
    tracked = np.flatnonzero(observed.any(axis=1))
    if len(tracked) < 7 or observed[tracked[0]:tracked[-1] + 1].any(axis=1).mean() < 0.70:
        return None
    # For two-hand signs keep the missing-hand masks and retain the coordination between sides.
    bilateral = active.all()
    if bilateral and observed[tracked[0]:tracked[-1] + 1].all(axis=1).mean() < 0.45:
        return None

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
    cost = (np.minimum(delta, 3) * together + 0.7 * different).sum(axis=-1)
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


def match_live_words(frames, aspect_ratio, entries):
    """Accept a reference only when the absolute match and lead over rivals are sufficient."""
    if not entries or (recording := _variants(frames, aspect_ratio)) is None:
        return None
    ranked = []
    for entry_id, video_id in entries:
        path = reference_path(video_id)
        if not path.is_file():
            continue
        reference = _reference_variants(path)
        if reference is not None:
            ranked.append((_distance(recording, reference), entry_id))
    ranked.sort()
    if not ranked or ranked[0][0] > MAX_DISTANCE:
        return None
    if len(ranked) > 1 and ranked[1][0] - ranked[0][0] < MIN_SEPARATION:
        return {"motion_detected": True, "status": "ambiguous_word"}
    distance, entry_id = ranked[0]
    return {"motion_detected": True, "status": "word_matched", "entry_id": entry_id, "match_method": "trajectory",
            "similarity": round(1.0 - distance, 3)}
