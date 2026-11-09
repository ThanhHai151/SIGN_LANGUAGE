import hashlib
import json
import os
from collections import Counter, defaultdict

file_path = "backend/data/diagnostics/pilot_pilot_20261108_0321.jsonl"

# Compute SHA256
h = hashlib.sha256()
with open(file_path, "rb") as f:
    while chunk := f.read(65536):
        h.update(chunk)
sha256 = h.hexdigest()
file_size = os.path.getsize(file_path)

print(f"File: {file_path}")
print(f"Size: {file_size:,} bytes")
print(f"SHA-256: {sha256}")

# Read and validate JSONL lines
total_lines = 0
invalid_json = 0
sessions = Counter()
trials = Counter()
expected_signs = Counter()
predicted_signs = Counter()
prediction_sources = Counter()
gating_statuses = Counter()
timestamps = []

records_by_trial = defaultdict(list)
corrupted_frames = []

with open(file_path, "r", encoding="utf-8") as f:
    for line_idx, line in enumerate(f, start=1):
        line = line.strip()
        if not line:
            continue
        total_lines += 1
        try:
            data = json.loads(line)
        except Exception as e:
            invalid_json += 1
            corrupted_frames.append((line_idx, str(e)))
            continue

        sess = data.get("session_id")
        sessions[sess] += 1

        trial = data.get("trial_id")
        trials[trial] += 1
        records_by_trial[trial].append((line_idx, data))

        exp = data.get("expected_sign")
        expected_signs[exp] += 1

        pred = data.get("pred_sign")
        predicted_signs[pred] += 1

        src = data.get("prediction_source")
        prediction_sources[src] += 1

        gate = data.get("gating_status")
        gating_statuses[gate] += 1

        ts = data.get("timestamp")
        if ts:
            timestamps.append(ts)

print(f"\nTotal JSON lines: {total_lines}")
print(f"Invalid JSON: {invalid_json}")
print(f"\nSessions: {dict(sessions)}")
print(f"Timestamps: min={min(timestamps) if timestamps else None}, max={max(timestamps) if timestamps else None}")
print(f"\nExpected Signs: {dict(expected_signs)}")
print(f"Predicted Signs: {dict(predicted_signs)}")
print(f"Prediction Sources: {dict(prediction_sources)}")
print(f"Gating Statuses: {dict(gating_statuses)}")
print(f"\nTotal Trials: {len(trials)}")
for tr, count in sorted(trials.items()):
    exp_in_tr = set(d.get("expected_sign") for _, d in records_by_trial[tr])
    frames = [d.get("frame_idx") for _, d in records_by_trial[tr]]
    min_fr, max_fr = min(frames) if frames else (None, None)
    dup_fr = len(frames) - len(set(frames))
    print(f"  Trial {tr}: {count} frames, expected={exp_in_tr}, frame_idx range [{min_fr}..{max_fr}], duplicates={dup_fr}")

