"""Prepare legacy skeletons or labelled videos for SSTCN training.

Videos: INPUT/sign_000/clip.mp4 ... INPUT/sign_101/clip.mp4.
Legacy skeletons: download with fetch_sstcn_data, then use --input .../raw.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np

from backend.training.sstcn_data import (
    CLASS_NAMES, DATA_ROOT, PREPROCESSING, canonical_sequence, class_mapping,
    grouped_split, load_source, sequence_hash, write_json,
)
from backend.training.fetch_sstcn_data import sha256

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def reconcile_groups(records):
    """Merge recording aliases and quarantine every sibling of a known test duplicate."""
    parents, hash_groups = {}, {}
    def find(group):
        parents.setdefault(group, group)
        while parents[group] != group:
            parents[group] = parents[parents[group]]
            group = parents[group]
        return group
    for r in records:
        if r["split"] == "test":
            continue
        key = (r["label"], r["group"])
        root = find(key)
        digest = (r["label"], r["sha256"])
        if digest in hash_groups:
            other = find(hash_groups[digest])
            parents[max(root, other)] = min(root, other)
        else:
            hash_groups[digest] = key
    for r in records:
        if r["split"] != "test":
            r["source_group"] = r["group"]
            r["group"] = find((r["label"], r["group"]))[1]
    test_hashes = {r["sha256"] for r in records if r["split"] == "test"}
    return {(r["label"], r["group"]) for r in records
            if r["split"] != "test" and r["sha256"] in test_hashes}


def video_to_landmarks(path, frame_stride=2):
    import cv2
    try:
        import mediapipe as mp
    except ImportError as e:
        raise RuntimeError("Install backend/training/requirements-preprocessing.txt in a separate venv") from e
    if not hasattr(mp, "solutions"):
        raise RuntimeError("The legacy Holistic extractor requires mediapipe==0.10.21")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames, frame_index = [], 0
    try:
        with mp.solutions.holistic.Holistic(model_complexity=1, min_detection_confidence=0.5,
                                           min_tracking_confidence=0.5) as detector:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_index += 1
                if frame_index % frame_stride:
                    continue
                result = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                parts = []
                for field, count in (("pose_landmarks", 33), ("left_hand_landmarks", 21),
                                     ("right_hand_landmarks", 21)):
                    landmarks = getattr(result, field)
                    parts.append(np.asarray([[lm.x, lm.y, lm.z] for lm in landmarks.landmark],
                                            dtype=np.float32) if landmarks else np.zeros((count, 3), np.float32))
                frames.append(np.concatenate(parts))
    finally:
        cap.release()
    if not frames:
        raise ValueError("Video contains no sampled frames")
    return np.asarray(frames), {"fps": fps if np.isfinite(fps) else None,
                                "decoded_frames": frame_index, "sampled_frames": len(frames),
                                "frame_stride": frame_stride, "extractor": "mediapipe.solutions.holistic",
                                "extractor_version": str(mp.__version__), "opencv_version": str(cv2.__version__),
                                "model_complexity": 1, "detection_confidence": 0.5, "tracking_confidence": 0.5}


def identity(path, root, embedded_label):
    relative = path.relative_to(root)
    match = next((re.fullmatch(r"sign_(\d{1,3})", part) for part in relative.parts
                  if re.fullmatch(r"sign_(\d{1,3})", part)), None)
    label = int(match.group(1)) if match else embedded_label
    if isinstance(label, bool) or not isinstance(label, (int, np.integer)) or not 0 <= int(label) < 102:
        raise ValueError("Missing/invalid class index: use sign_000 ... sign_101 folders")
    label = int(label)
    if embedded_label is not None:
        if isinstance(embedded_label, bool) or not isinstance(embedded_label, (int, np.integer)):
            raise ValueError("Embedded label must be an integer")
        if int(embedded_label) != label:
            raise ValueError("Folder label and embedded label disagree")
    # Historical _00 is the original; _01/_02/_03 are its augmented siblings.
    legacy = re.fullmatch(r"(sign_\d+_vid_\d+)_(\d+)", path.stem)
    group = f"{label}/{legacy.group(1) if legacy else path.stem}"
    augmentation = int(legacy.group(2)) if legacy else 0
    split = "test" if any(part.lower() in {"test", "test_set"} for part in relative.parts[:-1]) else "pending"
    return label, group, augmentation, split


def prepare(input_dir, output_dir, val_fraction=0.15, seed=42, frame_stride=2, overwrite=False, include_archives=None):
    input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).resolve()
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"{output_dir} exists; pass --overwrite to replace a generated dataset")
    if output_dir.exists() and not (output_dir / "manifest.json").is_file():
        raise ValueError("Refusing to replace a directory that is not a generated SSTCN dataset")
    if output_dir == input_dir or input_dir.is_relative_to(output_dir) or output_dir.is_relative_to(input_dir):
        raise ValueError("Input and output directories must be separate")
    paths = sorted(p for p in input_dir.rglob("*") if p.is_file()
                   and (p.suffix.lower() in VIDEO_EXTENSIONS or p.suffix.lower() == ".npy")
                   and "__MACOSX" not in p.parts
                   and (not include_archives or p.relative_to(input_dir).parts[0] in include_archives))
    if not paths:
        raise ValueError("No labelled .npy files or videos were found")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix="sstcn-prepare-", dir=output_dir.parent))
    records, rejected = [], []
    try:
        # Staging avoids holding the entire multi-GB dataset in RAM.
        staging = np.lib.format.open_memmap(temp / "staging.npy", mode="w+", dtype=np.float32,
                                            shape=(len(paths), 3, 30, 75))
        for p in paths:
            try:
                if p.suffix.lower() == ".npy":
                    value, embedded = load_source(p)
                    metadata = {"kind": "upstream_skeleton", "source_shape": list(np.asarray(value).shape)}
                else:
                    value, metadata = video_to_landmarks(p, frame_stride)
                    metadata["kind"] = "video"
                    embedded = None
                label, group, augmentation, split = identity(p, input_dir, embedded)
                sequence = canonical_sequence(value)
                row = len(records)
                staging[row] = sequence
                records.append({"row": row, "source": p.relative_to(input_dir).as_posix(),
                                "label": label, "group": group, "augmentation": augmentation,
                                "split": split, "sha256": sequence_hash(sequence), **metadata})
            except (ValueError, OSError, EOFError, KeyError, TypeError) as e:
                rejected.append({"source": p.relative_to(input_dir).as_posix(), "reason": str(e)})
        if not records:
            raise ValueError(f"No valid samples; first errors: {rejected[:3]}")
        # Identical recordings under different names must not cross split boundaries.
        blocked_groups = reconcile_groups(records)
        excluded_test_groups = [r["source"] for r in records
                                if r["split"] != "test" and (r["label"], r["group"]) in blocked_groups]
        # Keep supplied test samples. Remove exact duplicates from training first.
        seen, kept, duplicates = {}, [], []
        for r in sorted(records, key=lambda r: (r["split"] != "test", r["source"])):
            key = r["sha256"]
            if key in seen:
                original = seen[key]
                if original["label"] != r["label"]:
                    raise ValueError(f"Identical skeleton has conflicting labels: {original['source']} and {r['source']}")
                duplicates.append({"removed": r["source"], "kept": original["source"],
                                   "cross_test_boundary": r["split"] != original["split"]})
                continue
            seen[key] = r
            kept.append(r)
        kept = [r for r in kept if r["split"] == "test" or (r["label"], r["group"]) not in blocked_groups]
        pending = [r for r in kept if r["split"] != "test"]
        if pending:
            grouped_split(kept, val_fraction, seed)
        # Evaluate on original recordings only, not precomputed augmentations.
        excluded_aug = [r["source"] for r in kept if r["split"] != "train" and r["augmentation"] != 0]
        kept = sorted((r for r in kept if r["split"] == "train" or r["augmentation"] == 0), key=lambda r: r["source"])
        x = np.lib.format.open_memmap(temp / "sequences.npy", mode="w+", dtype=np.float32,
                                     shape=(len(kept), 3, 30, 75))
        labels = []
        for i, r in enumerate(kept):
            x[i] = staging[r["row"]]
            r["row"] = i
            labels.append(r["label"])
        x.flush()
        del x, staging
        (temp / "staging.npy").unlink()
        np.save(temp / "labels.npy", np.asarray(labels, dtype=np.int64), allow_pickle=False)
        counts = {split: dict(sorted(Counter(r["label"] for r in kept if r["split"] == split).items()))
                  for split in ("train", "val", "test")}
        manifest = {"schema_version": 1, "class_names": CLASS_NAMES, "preprocessing": PREPROCESSING,
                    "seed": seed, "val_fraction": val_fraction, "split_method": "stratified_source_video_groups",
                    "identity_limitations": "Source filenames identify recording groups; signer identity is unavailable. External test IDs may restart numbering.",
                    "input_root": str(input_dir), "counts": {s: sum(c.values()) for s, c in counts.items()},
                    "included_archives": include_archives,
                    "per_class_counts": counts, "rejected": rejected, "duplicates": duplicates,
                    "excluded_test_overlap_groups": excluded_test_groups,
                    "excluded_eval_augmentations": excluded_aug, "samples": kept}
        manifest["files"] = {name: sha256(temp / name) for name in ("sequences.npy", "labels.npy")}
        write_json(temp / "manifest.json", manifest)
        write_json(temp / "classes.json", class_mapping())
        if output_dir.exists():
            shutil.rmtree(output_dir)
        temp.replace(output_dir)
        print(json.dumps({"output": str(output_dir), "counts": manifest["counts"],
                          "rejected": len(rejected), "duplicates": len(duplicates)}, ensure_ascii=False), flush=True)
        return manifest
    finally:
        if temp.exists():
            shutil.rmtree(temp)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DATA_ROOT / "raw")
    p.add_argument("--output", type=Path, default=DATA_ROOT / "prepared")
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--frame-stride", type=int, default=2)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--include-archives", nargs="+", help="Explicit subset of top-level archive folders; recorded in the manifest")
    a = p.parse_args()
    if a.frame_stride < 1:
        p.error("--frame-stride must be positive")
    prepare(a.input, a.output, a.val_fraction, a.seed, a.frame_stride, a.overwrite, a.include_archives)


if __name__ == "__main__":
    main()
