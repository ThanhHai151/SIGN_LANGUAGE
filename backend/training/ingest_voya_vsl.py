"""Ingestion adapter for Hugging Face `Kateht/VOYA_VSL` MediaPipe keypoint dataset.

Converts `.npz` MediaPipe Holistic sequences (e.g. shape [60, 1605] or [T, 1662] or [T, 225])
into canonical raw [30, 75, 3] (33 pose + 21 left hand + 21 right hand) sequences
and emits a `records.jsonl` manifest compatible with `prepare_unified_dataset.py`.

Usage:
  python -m backend.training.ingest_voya_vsl --output backend/data/vsl_unified/sources/voya_vsl
  python -m backend.training.ingest_voya_vsl --local-dir /path/to/voya_vsl --output backend/data/vsl_unified/sources/voya_vsl
"""

import argparse
import json
from pathlib import Path

import numpy as np

from backend.vision.skeleton_norm import resample_sequence_linear

POSE_N = 33
HAND_N = 21
TARGET_JOINTS = 75


def extract_75_joints_from_voya_array(arr: np.ndarray) -> np.ndarray:
    """Map VOYA_VSL / MediaPipe Holistic flattened arrays into [T, 75, 3] (pose + lh + rh).

    Supported shapes:
    - [T, 75, 3] or [T, 225]: direct pose(33) + lh(21) + rh(21) xyz
    - [T, 1662]: MediaPipe Holistic standard = pose(33*4=132) + face(468*3=1404) + lh(21*3=63) + rh(21*3=63)
    - [T, 1629]: MediaPipe Holistic xyz-only = pose(33*3=99) + face(468*3=1404) + lh(63) + rh(63)
    - [T, 1605]: VOYA_VSL layout = pose(25*3=75) or pose(33*3=99) + reduced face + lh(63) + rh(63).
      In all Holistic flat layouts, the final 126 dims are left_hand(63) + right_hand(63),
      and the leading dims are pose (33*3=99 or 25*3=75).
    - [T, 195]: KVSL 65-joint layout = pose(23*3=69) + lh(63) + rh(63)
    """
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 3 and a.shape[1:] == (75, 3):
        return a
    if a.ndim == 3 and a.shape[1:] == (65, 3):
        out = np.zeros((a.shape[0], 75, 3), dtype=np.float32)
        out[:, :23, :] = a[:, :23, :]
        out[:, 33:54, :] = a[:, 23:44, :]
        out[:, 54:75, :] = a[:, 44:65, :]
        return out
    if a.ndim != 2 or a.shape[0] == 0:
        raise ValueError(f"Unsupported VOYA array shape: {a.shape}")

    T, D = a.shape
    out = np.zeros((T, TARGET_JOINTS, 3), dtype=np.float32)

    if D == 225:
        return a.reshape(T, 75, 3)
    if D == 195:
        out[:, :23, :] = a[:, :69].reshape(T, 23, 3)
        out[:, 33:54, :] = a[:, 69:132].reshape(T, 21, 3)
        out[:, 54:75, :] = a[:, 132:195].reshape(T, 21, 3)
        return out
    if D == 1662:
        # pose has (x, y, z, visibility) -> take first 3 channels
        pose = a[:, :132].reshape(T, 33, 4)[:, :, :3]
        lh = a[:, -126:-63].reshape(T, 21, 3)
        rh = a[:, -63:].reshape(T, 21, 3)
        out[:, :33, :] = pose
        out[:, 33:54, :] = lh
        out[:, 54:75, :] = rh
        return out
    if D == 1629:
        out[:, :33, :] = a[:, :99].reshape(T, 33, 3)
        out[:, 33:54, :] = a[:, -126:-63].reshape(T, 21, 3)
        out[:, 54:75, :] = a[:, -63:].reshape(T, 21, 3)
        return out
    if D == 1605:
        # VOYA_VSL: 25 upper-body pose*3 (75) + 468 face*3 (1404) + lh(63) + rh(63) = 1605
        # OR 33 pose*3 (99) + 460 face*3 (1380) + lh(63) + rh(63) = 1605
        # Leading 23 upper-body pose landmarks (indices 0..22 -> 69 dims) are identical in both!
        # Lower-body pose (23..32) is zeroed in Preprocessing v2 anyway.
        out[:, :23, :] = a[:, :69].reshape(T, 23, 3)
        out[:, 33:54, :] = a[:, -126:-63].reshape(T, 21, 3)
        out[:, 54:75, :] = a[:, -63:].reshape(T, 21, 3)
        return out

    raise ValueError(f"Unsupported flattened feature dimension D={D} (shape={a.shape})")


def _load_npz_sequences(path: Path):
    with np.load(path, allow_pickle=False) as data:
        keys = list(data.keys())
        seq_key = None
        for candidate in ("sequences", "sequence", "keypoints", "data", "x", "arr_0"):
            if candidate in data:
                seq_key = candidate
                break
        if seq_key is None and keys:
            seq_key = keys[0]
        if seq_key is None:
            raise ValueError(f"Empty npz file: {path}")
        seq = data[seq_key]
        if seq_key == "sequences" and seq.ndim in (3, 4):
            labels = data["labels"] if "labels" in data else [None] * len(seq)
            if len(labels) != len(seq):
                raise ValueError("VOYA sequences and labels have different lengths")
            for sample, label in zip(seq, labels):
                yield sample, None if label is None else str(np.asarray(label).item())
            return
        label = None
        if "label" in data:
            val = data["label"]
            label = str(val.item() if val.ndim == 0 else val[0])
        yield seq, label


def ingest_from_directory(local_dir: Path, output_dir: Path, target_len: int = 30) -> dict:
    local_dir = Path(local_dir)
    output_dir = Path(output_dir)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # Load optional labels.json mapping id -> label
    id_to_label = {}
    for cand in (local_dir / "labels.json", local_dir / "label_map.json"):
        if cand.exists():
            raw = json.loads(cand.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                # Support both {"label": idx} and {"idx": "label"}
                first_k, first_v = next(iter(raw.items()))
                if isinstance(first_v, int):
                    id_to_label = {str(v): str(k) for k, v in raw.items()}
                else:
                    id_to_label = {str(k): str(v) for k, v in raw.items()}
            elif isinstance(raw, list):
                id_to_label = {str(i): str(v) for i, v in enumerate(raw)}
            break

    records = []
    errors = []
    npz_files = sorted(local_dir.rglob("*.npz"))
    for idx, npz_path in enumerate(npz_files):
        try:
            relative = npz_path.relative_to(local_dir).as_posix()
            for row, (raw_arr, inner_label) in enumerate(_load_npz_sequences(npz_path)):
                try:
                    seq_75 = np.nan_to_num(extract_75_joints_from_voya_array(raw_arr), nan=0, posinf=0, neginf=0)
                    if not np.any(seq_75):
                        raise ValueError("All-zero sequence")
                    seq_30 = resample_sequence_linear(seq_75, target_len=target_len)
                except ValueError as e:
                    errors.append({"path": relative, "row": row, "error": str(e)})
                    continue

                raw_label = id_to_label.get(npz_path.stem)
                if raw_label is None:
                    label_id = inner_label or npz_path.parent.name
                    raw_label = id_to_label.get(label_id, label_id)
                out_name = f"voya_{idx:06d}_{row:05d}.npy"
                np.save(samples_dir / out_name, seq_30, allow_pickle=False)
                records.append({
                    "file": f"samples/{out_name}",
                    "raw_label": raw_label,
                    "source_name": "voya_vsl",
                    "original_path": relative,
                    "original_row": row,
                    "group": f"voya/{relative}/{row}",
                    "signer_id": None,
                    "view": "front",
                    "video_id": None,
                })
        except Exception as e:
            errors.append({"path": str(npz_path), "error": str(e)})

    manifest_path = output_dir / "records.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {
        "source": "Kateht/VOYA_VSL",
        "total_found": len(npz_files),
        "ingested": len(records),
        "errors": len(errors),
        "records_file": str(manifest_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "errors.json").write_text(json.dumps(errors[:500], ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def download_voya_snapshot(dest_dir: Path, repo_id: str = "Kateht/VOYA_VSL") -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise RuntimeError(
            "Cần cài `huggingface_hub` (`pip install huggingface_hub`) hoặc truyền `--local-dir` đã tải sẵn."
        ) from e
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=str(dest_dir))
    return Path(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--local-dir", type=Path, default=None, help="Thư mục VOYA_VSL đã tải sẵn (bỏ qua bước tải HF)")
    p.add_argument("--download-dir", type=Path, default=Path("backend/data/vsl_unified/raw_downloads/voya_vsl"))
    p.add_argument("--repo-id", default="Kateht/VOYA_VSL")
    p.add_argument("--output", type=Path, default=Path("backend/data/vsl_unified/sources/voya_vsl"))
    args = p.parse_args()

    src_dir = args.local_dir if args.local_dir else download_voya_snapshot(args.download_dir, repo_id=args.repo_id)
    summary = ingest_from_directory(src_dir, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
