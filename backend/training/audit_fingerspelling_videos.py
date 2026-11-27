"""Replay QIPEDC letter and digit RGB files through the camera landmark recognizer."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np

from backend.models.lexicon import motion_clip
from backend.models.video_cache import ensure_cached
from backend.training.ingest_videos import _import_mediapipe_holistic
from backend.training.sstcn_data import write_json

DATA = Path(__file__).resolve().parents[1] / "data"


def sources():
    letter_meta = json.loads((DATA / "alphabet_v2/metadata/videos_metadata.json").read_text())
    active_clips = json.loads((DATA / "letter_clips.json").read_text())
    for label, item in letter_meta.items():
        yield label, "letters", item["video_id"], active_clips[label.lower()]["clip"]
    for digit in range(11):
        video_id = "D0535N" if digit == 6 else f"D{529 + digit:04}"
        yield str(digit), "digits", video_id, motion_clip(video_id)


def locate(item, download):
    label, mode, video_id, interval = item
    path = DATA / "media_cache/videos" / f"{video_id}.mp4"
    if not path.is_file() and download:
        ensure_cached("videos", video_id, timeout=12)
    return label, mode, video_id, interval, path


def extract(item):
    label, mode, video_id, interval, path = item
    row = {"label": label, "mode": mode, "video": video_id}
    if not path.is_file():
        return {**row, "error": "video_missing"}
    import cv2

    _import_mediapipe_holistic()  # Load optional legacy MediaPipe and its drawing dependencies.
    import mediapipe as mp

    cv2.setNumThreads(1)
    video = cv2.VideoCapture(str(path))
    points, times, sides = [], [], []
    fps = video.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 1:
        video.release()
        return {**row, "error": "invalid_video"}
    try:
        with mp.solutions.hands.Hands(
            model_complexity=1, max_num_hands=1, min_detection_confidence=0.35, min_tracking_confidence=0.30,
        ) as detector:
            frame_index = 0
            while True:
                available, image = video.read()
                if not available:
                    break
                if frame_index % 2 == 0:
                    height, width = image.shape[:2]
                    image = cv2.resize(image, (640, round(640 * height / width)), interpolation=cv2.INTER_AREA)
                    detected = detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    if detected.multi_hand_landmarks:
                        points.append([[p.x, p.y, p.z] for p in detected.multi_hand_landmarks[0].landmark])
                        sides.append("Left" if detected.multi_handedness[0].classification[0].label == "Right" else "Right")
                    else:
                        points.append(None)
                        sides.append(None)
                    times.append(frame_index / fps)
                frame_index += 1
    finally:
        video.release()
    if not points:
        return {**row, "error": "video_empty"}
    return {**row, "landmarks": points, "timestamps": times, "handedness": sides,
            "active_interval_sec": list(interval or (0, times[-1]))}


def run(output, workers, download):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        inputs = list(pool.map(lambda item: locate(item, download), sources()))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        extracted = list(pool.map(extract, inputs))
    from backend.vision.recognizer import VSLRecognizer

    recognizer = VSLRecognizer()
    results = []
    for item in extracted:
        if "error" in item:
            results.append(item)
            continue
        recognizer.reset()
        frame_counts = Counter()
        total_frames = len(item["landmarks"])
        tracked_frames = 0
        start, end = item["active_interval_sec"]
        for landmarks, timestamp, side in zip(
            item.pop("landmarks"), item.pop("timestamps"), item.pop("handedness"),
        ):
            tracked_frames += int(landmarks is not None)
            hands = [{"landmarks": landmarks, "handedness": side}] if landmarks else []
            result = recognizer.predict_landmarks(
                landmarks, multi_hands=hands, timestamp=timestamp + 100.0,
                session_id=item["video"], mode=item["mode"],
            )
            if start <= timestamp <= end + 0.24 and result["sign"] != "NEUTRAL":
                frame_counts[result["sign"]] += 1
        captions = list(recognizer.sentence_tokens)
        results.append({
            **item, "total_frames": total_frames, "tracked_frames": tracked_frames,
            "active_predictions": dict(frame_counts), "final_caption": captions,
            "correct_caption": captions == [item["label"]],
        })
    summary = {}
    for mode in ("letters", "digits"):
        rows = [item for item in results if item["mode"] == mode]
        tested = [item for item in rows if "error" not in item]
        correct = sum(item["correct_caption"] for item in tested)
        summary[mode] = {
            "correct": correct, "tested": len(tested), "expected": len(rows),
            "accuracy": correct / len(tested) if tested else None,
        }
    write_json(output, {"summary": summary, "videos": results})
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    run(args.output, args.workers, args.download)
