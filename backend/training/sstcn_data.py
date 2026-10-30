"""Shared SSTCN data contract: raw MediaPipe xyz, pose + left + right hand.

Canonical storage is float32 [N, 3, 30, 75]. No per-sample normalization is
applied: the deployed historical recognizer consumes image-normalized xyz.
"""
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from backend.models.vsl_102_dictionary import SIGN_DICT_102

CLASS_NAMES = [SIGN_DICT_102[i] for i in range(102)]
DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "sstcn"
PREPROCESSING = {
    "version": 1, "shape": [3, 30, 75], "dtype": "float32",
    "joint_order": ["pose:0..32", "left_hand:0..20", "right_hand:0..20"],
    "coordinates": "mediapipe_image_normalized_xyz",
    "missing_landmarks": "zero", "normalization": "none",
    "temporal_sampling": "linear_resample_to_30_after_frame_stride",
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def class_mapping():
    return {"classes": [{"index": i, "label": label} for i, label in enumerate(CLASS_NAMES)],
            "unknown": {"index": 102, "is_model_class": False, "label": "UNKNOWN"}}


class _NumpyUnpickler(pickle.Unpickler):
    """Read the source's scalar object-array dictionaries without arbitrary globals."""
    def find_class(self, module, name):
        try:
            from numpy._core import multiarray
        except ImportError:  # NumPy 1.x in the isolated legacy MediaPipe environment
            from numpy.core import multiarray
        allowed = {("numpy", "ndarray"): np.ndarray, ("numpy", "dtype"): np.dtype}
        for mod in ("numpy.core.multiarray", "numpy._core.multiarray"):
            allowed[(mod, "_reconstruct")] = multiarray._reconstruct
            allowed[(mod, "scalar")] = multiarray.scalar
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"Forbidden pickle global: {module}.{name}")
        return allowed[(module, name)]


def load_source(path):
    """Support numeric .npy and legacy {'keypoints': array, 'label': int} .npy."""
    with Path(path).open("rb") as f:
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            _, _, dtype = np.lib.format.read_array_header_1_0(f)
        elif version == (2, 0):
            _, _, dtype = np.lib.format.read_array_header_2_0(f)
        else:
            raise ValueError(f"Unsupported NPY header: {version}")
        if dtype.hasobject:
            value = _NumpyUnpickler(f).load()
            if not isinstance(value, np.ndarray) or value.shape != ():
                raise ValueError("Expected a scalar object array containing a sample dictionary")
            value = value.item()
            if not isinstance(value, dict) or "keypoints" not in value:
                raise ValueError("Object sample must contain keypoints")
            return value["keypoints"], value.get("label")
    return np.load(path, allow_pickle=False), None


def canonical_sequence(value):
    a = np.asarray(value)
    if a.ndim == 2 and a.shape[1] == 225:
        a = a.reshape(-1, 75, 3)
    elif a.ndim == 3 and a.shape[0] == 3 and a.shape[2] == 75:
        a = a.transpose(1, 2, 0)
    if a.ndim != 3 or a.shape[1:] != (75, 3) or len(a) == 0:
        raise ValueError(f"Expected [T,75,3], [3,T,75] or [T,225]; received {a.shape}")
    a = np.asarray(a, dtype=np.float32)
    if not np.isfinite(a).all():
        raise ValueError("Non-finite landmark coordinates")
    if not np.any(a):
        raise ValueError("All landmarks are missing")
    if len(a) != 30:
        positions = np.linspace(0, len(a) - 1, 30)
        lo = positions.astype(int)
        hi = np.minimum(lo + 1, len(a) - 1)
        weight = (positions - lo).astype(np.float32)[:, None, None]
        a = a[lo] * (1 - weight) + a[hi] * weight
    return np.ascontiguousarray(a.transpose(2, 0, 1), dtype=np.float32)


def sequence_hash(sequence):
    return hashlib.sha256(np.asarray(sequence, dtype="<f4").tobytes()).hexdigest()


def grouped_split(records, val_fraction=0.15, seed=42):
    """Stratify source-video groups per class. Augmented siblings stay together."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    rng = np.random.default_rng(seed)
    for label in sorted({r["label"] for r in records if r["split"] != "test"}):
        selected = [r for r in records if r["label"] == label and r["split"] != "test"]
        groups = sorted({r["group"] for r in selected if r.get("augmentation", 0) == 0})
        if len(groups) < 2:
            raise ValueError(f"Class {label} needs at least two source-video groups with originals for train/val")
        rng.shuffle(groups)
        n = max(1, min(len(groups) - 1, round(len(groups) * val_fraction)))
        val_groups = set(groups[:n])
        for r in selected:
            r["split"] = "val" if r["group"] in val_groups else "train"
    return records


def load_prepared(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["class_names"] != CLASS_NAMES or manifest["preprocessing"] != PREPROCESSING:
        raise ValueError("Dataset class order or preprocessing contract does not match SSTCN")
    x = np.load(directory / "sequences.npy", mmap_mode="r", allow_pickle=False)
    y = np.load(directory / "labels.npy", allow_pickle=False)
    records = manifest["samples"]
    from backend.training.fetch_sstcn_data import sha256
    for filename, digest in manifest.get("files", {}).items():
        if sha256(directory / filename) != digest:
            raise ValueError(f"Prepared dataset checksum mismatch: {filename}")
    if x.shape != (len(records), 3, 30, 75) or y.shape != (len(records),):
        raise ValueError("Prepared arrays do not match the manifest")
    if any(int(y[i]) != r["label"] or r["row"] != i for i, r in enumerate(records)):
        raise ValueError("Dataset labels/rows do not match the manifest")
    return x, y, manifest
