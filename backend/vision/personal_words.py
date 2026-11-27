"""Explicitly labelled live word examples, scoped to one calibration profile."""

from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

import numpy as np

from backend.vision import qipedc_live
from backend.vision.qipedc_practice import analyze_live_words


def save_reference(root, entry_id, sequence, aspect_ratio):
    if not np.isfinite(sequence).all() or not 10 <= len(sequence) <= 180:
        raise ValueError("Mẫu cần từ 10 đến 180 khung hình hợp lệ.")
    if not analyze_live_words(sequence, aspect_ratio)["motion_detected"]:
        raise ValueError("Mẫu cần có chuyển động rõ và thấy hai vai cùng bàn tay.")
    variants = qipedc_live._variants(sequence, aspect_ratio)
    if variants is None:
        raise ValueError("Mẫu bị mất dấu tay quá lâu. Hãy ghi lại.")
    folder = Path(root) / entry_id
    folder.mkdir(parents=True, exist_ok=True)
    features, observed, penalties = variants
    fingerprint = sha256(features.tobytes() + observed.tobytes() + penalties.tobytes()).hexdigest()
    temporary = folder / f".{uuid4().hex}.npz"
    target = folder / f"{fingerprint}.npz"
    try:
        np.savez_compressed(temporary, features=features, observed=observed, penalties=penalties)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return len(list(folder.glob("[!.]*.npz")))


@lru_cache(maxsize=128)
def _read_reference(path, mtime):
    del mtime
    with np.load(path, allow_pickle=False) as data:
        return data["features"], data["observed"], data["penalties"]


def load_references(root, allowed_ids):
    folder = Path(root)
    if not folder.is_dir():
        return {}
    result = {}
    for word_dir in folder.iterdir():
        if word_dir.name not in allowed_ids or not word_dir.is_dir():
            continue
        samples = []
        paths = []
        for path in word_dir.glob("[!.]*.npz"):
            try:
                paths.append((path.stat().st_mtime_ns, path))
            except OSError:
                continue
        for mtime, path in sorted(paths, reverse=True)[:5]:
            try:
                samples.append(_read_reference(path, mtime))
            except (ValueError, OSError, KeyError):
                continue
        if samples:
            result[word_dir.name] = samples
    return result
