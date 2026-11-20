"""Multi-source VSL Video -> 75-joint MediaPipe Holistic Skeleton Ingestion.

Supports:
- `visign`: Hugging Face `namwu/ViSignLanguage-Video` (label subfolders or `dataset_metadata.csv`)
- `qipedc`: Kaggle `VSL-Vietnamese Sign Languages` / QIPEDC dictionary videos
            (automatically uses `active: [t_start, t_end]` from `backend/data/motions/<video_id>.json`
            to trim idle standing frames!)
- `vsl400`: Zenodo `VSL400` / `Multi-VSL` multi-view clips (front/left/right, preserves signer_id)
- `folder`: Generic `<input_dir>/<label>/*.mp4` structure

Outputs raw [30, 75, 3] `.npy` samples + `records.jsonl` for `prepare_unified_dataset.py`.
"""

import argparse
import csv
import hashlib
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from backend.models.lexicon import motion_clip, norm_key
from backend.vision.skeleton_norm import resample_sequence_linear

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
_VIEW_RE = re.compile(r"(front|left|right|cam0|cam1|cam2|v1|v2|v3)", re.IGNORECASE)
_SIGNER_RE = re.compile(r"(?:signer|s|p|user)[_\-]?(\d+)", re.IGNORECASE)


def extract_video_to_75_joints(
    video_path: Path,
    target_len: Optional[int] = 30,
    frame_stride: int = 2,
    active_window: Optional[Tuple[float, float]] = None,
    webcam_profile: bool = False,
) -> np.ndarray:
    """Extract [target_len, 75, 3] MediaPipe Holistic keypoints (33 pose + 21 lh + 21 rh)."""
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    if fps <= 1.0 or not np.isfinite(fps):
        fps = 25.0

    start_frame, end_frame = 0, None
    if active_window and len(active_window) == 2 and active_window[1] > active_window[0]:
        start_frame = max(0, int(active_window[0] * fps))
        end_frame = max(start_frame + 4, int(active_window[1] * fps))

    frames_kps = []
    mp_holistic = mp.solutions.holistic
    options = dict(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5,
                   min_tracking_confidence=0.5)
    if webcam_profile:
        options.update(model_complexity=0, smooth_landmarks=False, min_detection_confidence=0.35,
                       min_tracking_confidence=0.30)
    with mp_holistic.Holistic(**options) as holistic:
        frame_idx = 0
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_idx = start_frame

        while cap.isOpened():
            if end_frame is not None and frame_idx > end_frame:
                break
            ret, frame = cap.read()
            if not ret:
                break
            if (frame_idx - start_frame) % max(1, frame_stride) == 0:
                if webcam_profile:
                    height, width = frame.shape[:2]
                    scale = min(1.0, 480 / max(width, height))
                    if scale < 1.0:
                        frame = cv2.resize(frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                res = holistic.process(rgb)

                pose = (
                    np.array([[lm.x, lm.y, lm.z] for lm in res.pose_landmarks.landmark], dtype=np.float32)
                    if res.pose_landmarks
                    else np.zeros((33, 3), dtype=np.float32)
                )
                lh = (
                    np.array([[lm.x, lm.y, lm.z] for lm in res.left_hand_landmarks.landmark], dtype=np.float32)
                    if res.left_hand_landmarks
                    else np.zeros((21, 3), dtype=np.float32)
                )
                rh = (
                    np.array([[lm.x, lm.y, lm.z] for lm in res.right_hand_landmarks.landmark], dtype=np.float32)
                    if res.right_hand_landmarks
                    else np.zeros((21, 3), dtype=np.float32)
                )
                landmarks = np.concatenate([pose, lh, rh], axis=0)
                if webcam_profile:
                    # MediaPipe uses image width for x/z and height for y.
                    landmarks[:, 1] *= frame.shape[0] / frame.shape[1]
                frames_kps.append(landmarks)
            frame_idx += 1

    cap.release()
    if len(frames_kps) < 4:
        raise ValueError(f"Too few extracted frames ({len(frames_kps)}) in {video_path}")

    arr = np.asarray(frames_kps, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.any(arr[:, 33:, :]):
        raise ValueError(f"No hand landmarks detected in {video_path}")
    return arr if target_len is None else resample_sequence_linear(arr, target_len=target_len)


def discover_videos(input_dir: Path, source_type: str, csv_path: Optional[Path] = None) -> List[Dict]:
    """Discover video files and metadata according to source convention."""
    input_dir = Path(input_dir)
    items: List[Dict] = []

    # 1. CSV metadata mode (for ViSignLanguage-Video or Kaggle QIPEDC CSV)
    candidate_csvs = [csv_path] if csv_path else [
        input_dir / "dataset_metadata.csv",
        input_dir / "label.csv",
        input_dir / "data.csv",
        input_dir / "metadata.csv",
    ]
    used_csv = next((p for p in candidate_csvs if p and p.exists()), None)
    videos = sorted(p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
    if used_csv:
        by_name = {}
        for video in videos:
            by_name.setdefault(video.name, []).append(video)
        with used_csv.open("r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                low = {k.lower().strip(): (v or "").strip() for k, v in row.items() if k}
                rel_vid = low.get("video_path") or low.get("filepath") or low.get("path") or low.get("video") or low.get("file") or low.get("filename") or ""
                label = low.get("label") or low.get("word") or low.get("gloss") or low.get("class") or ""
                split = low.get("split", "").lower()
                if not rel_vid or not label:
                    continue
                path = Path(rel_vid.replace("\\", "/"))
                vid_path = input_dir / path
                if path.is_absolute() or not vid_path.is_file() or not vid_path.resolve().is_relative_to(input_dir.resolve()):
                    matches = by_name.get(path.name, [])
                    contextual = [p for p in matches if norm_key(p.parent.name) == norm_key(label) and split in p.parts]
                    matches = contextual or matches
                    if len(matches) != 1:
                        continue
                    vid_path = matches[0]
                vid_id = vid_path.stem
                signer = low.get("signer_id") or low.get("signer") or None
                view = low.get("view") or "front"
                items.append({
                    "video_path": vid_path,
                    "raw_label": label,
                    "video_id": vid_id,
                    "signer_id": signer,
                    "view": view,
                    "split": split,
                    "group": f"{source_type}/{low.get('group') or vid_id}",
                })
        if items:
            return items

    # 2. Subfolder mode: <root>/<label>/*.(mp4|avi|...)
    qipedc_labels = {}
    if source_type in {"qipedc", "kaggle_qipedc"}:
        dictionary = Path(__file__).resolve().parents[1] / "data" / "sources" / "qipedc_dictionary.json"
        qipedc_labels = {row[0].upper(): row[1] for row in json.loads(dictionary.read_text(encoding="utf-8"))}
    for p in videos:
        vid_id = p.stem
        raw_label = qipedc_labels.get(vid_id.upper(), p.parent.name)
        if qipedc_labels and vid_id.upper() not in qipedc_labels and p.parent.name.lower() == "videos":
            raise ValueError(f"Missing QIPEDC label for {vid_id}; supply a metadata CSV.")
        signer = None
        view = "front"
        if source_type == "vsl400":
            m_s = _SIGNER_RE.search(str(p))
            if m_s:
                signer = f"signer_{m_s.group(1)}"
            m_v = _VIEW_RE.search(str(p))
            if m_v:
                view = m_v.group(1).lower()
            # Group by signer + base clip stem (without view tag) so multi-view clips stay in same split!
            base_clip = _VIEW_RE.sub("", vid_id).strip("_-")
            group = f"vsl400/{raw_label}/{signer or 's0'}/{base_clip or vid_id}"
        else:
            group = f"{source_type}/{raw_label}/{vid_id}"
        items.append({
            "video_path": p,
            "raw_label": raw_label,
            "video_id": vid_id,
            "signer_id": signer,
            "view": view,
            "group": group,
            "split": "test" if "test" in p.relative_to(input_dir).parts[:-2] else "",
        })
    return items


def _worker_extract(task: Dict) -> Tuple[bool, Dict]:
    try:
        vid_path = Path(task["video_path"])
        active = None
        if task["use_motion_active"] and task.get("video_id"):
            clip = motion_clip(task["video_id"])
            if clip and len(clip) == 2 and clip[1] > clip[0]:
                active = (float(clip[0]), float(clip[1]))
        out_path = Path(task["out_npy"])
        if not (task["resume"] and out_path.is_file()):
            seq = extract_video_to_75_joints(
                vid_path,
                target_len=task["target_len"],
                frame_stride=task["frame_stride"],
                active_window=active,
            )
            temp = out_path.with_suffix(".part")
            with temp.open("wb") as output:
                np.save(output, seq.astype(np.float32), allow_pickle=False)
            temp.replace(out_path)
        return True, {
            "file": f"samples/{out_path.name}",
            "raw_label": task["raw_label"],
            "source_name": task["source_type"],
            "original_path": str(vid_path),
            "group": task["group"],
            "signer_id": task.get("signer_id"),
            "view": task.get("view", "front"),
            "video_id": task.get("video_id"),
            "split": task.get("split"),
            "trimmed_active": list(active) if active else None,
        }
    except Exception as e:
        return False, {"path": str(task["video_path"]), "error": str(e)}


def ingest_videos(
    input_dir: Path,
    output_dir: Path,
    source_type: str = "folder",
    csv_path: Optional[Path] = None,
    workers: int = 4,
    frame_stride: int = 2,
    max_videos: Optional[int] = None,
    resume: bool = False,
) -> dict:
    output_dir = Path(output_dir)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    discovered = discover_videos(input_dir, source_type=source_type, csv_path=csv_path)
    if max_videos is not None and max_videos > 0:
        discovered = discovered[:max_videos]

    tasks = []
    for idx, item in enumerate(discovered):
        sample_id = f"{idx:06d}"
        if resume:
            path = item["video_path"]
            stat = path.stat()
            identity = (path.relative_to(input_dir).as_posix(), stat.st_size, stat.st_mtime_ns, frame_stride)
            sample_id = hashlib.sha256(repr(identity).encode()).hexdigest()[:24]
        tasks.append({
            **item,
            "source_type": source_type,
            "target_len": 30,
            "frame_stride": frame_stride,
            "use_motion_active": source_type in ("qipedc", "kaggle_qipedc"),
            "out_npy": str(samples_dir / f"{source_type}_{sample_id}.npy"),
            "resume": resume,
        })

    records = []
    errors = []
    if workers <= 1 or len(tasks) <= 4:
        for t in tasks:
            ok, res = _worker_extract(t)
            (records if ok else errors).append(res)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for count, (ok, res) in enumerate(ex.map(_worker_extract, tasks, chunksize=8), start=1):
                (records if ok else errors).append(res)
                if resume and count % 250 == 0:
                    print(f"{source_type}: {count}/{len(tasks)} clips, {len(errors)} rejected", flush=True)

    records.sort(key=lambda r: r["file"])
    records_path = output_dir / "records.jsonl"
    with records_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "source_type": source_type,
        "input_dir": str(input_dir),
        "discovered": len(discovered),
        "ingested": len(records),
        "errors": len(errors),
        "records_file": str(records_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if errors:
        (output_dir / "errors.json").write_text(json.dumps(errors[:500], ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source", choices=["visign", "qipedc", "vsl400", "folder"], default="folder")
    p.add_argument("--csv", type=Path, default=None)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    p.add_argument("--frame-stride", type=int, default=2)
    p.add_argument("--max-videos", type=int, default=None)
    args = p.parse_args()

    summary = ingest_videos(
        input_dir=args.input_dir,
        output_dir=args.output,
        source_type=args.source,
        csv_path=args.csv,
        workers=args.workers,
        frame_stride=args.frame_stride,
        max_videos=args.max_videos,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
