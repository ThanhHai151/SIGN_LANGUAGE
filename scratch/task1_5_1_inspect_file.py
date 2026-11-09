import sys
import os
import json
import hashlib
import collections

filepath = "backend/data/diagnostics/pilot_pilot_20261108_183703.jsonl"

# 1. SHA256
h = hashlib.sha256()
with open(filepath, "rb") as f:
    while chunk := f.read(65536):
        h.update(chunk)
sha256 = h.hexdigest()
print(f"File: {filepath}")
print(f"File Size: {os.path.getsize(filepath)} bytes")
print(f"SHA256: {sha256}")

# 2. Parse lines
lines = []
parse_errors = []
with open(filepath, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            lines.append((idx, rec))
        except Exception as e:
            parse_errors.append((idx, str(e)))

print(f"Total lines parsed: {len(lines)}")
print(f"Parse errors: {len(parse_errors)}")
if parse_errors:
    print(f"First parse error: {parse_errors[0]}")

# 3. Session IDs and Trial IDs
sessions = collections.Counter(r.get("session_id") for _, r in lines)
trials = collections.Counter(r.get("trial_id") for _, r in lines)
expected_signs = collections.Counter(r.get("expected_sign") for _, r in lines)
final_signs = collections.Counter(r.get("final_sign") for _, r in lines)

print(f"\nSessions: {dict(sessions)}")
print(f"Expected signs: {dict(expected_signs)}")
print(f"Final signs: {dict(final_signs)}")
print(f"Total distinct trials: {len(trials)}")
print("Trials distribution:")
for t_id, cnt in sorted(trials.items()):
    print(f"  {t_id}: {cnt} frames")

# 4. Check first and last frames
first_f = lines[0][1]
last_f = lines[-1][1]
print("\nFirst frame keys:", list(first_f.keys()))
print(f"First frame timestamp: {first_f.get('timestamp')}, frame_idx: {first_f.get('frame_idx')}, trial_id: {first_f.get('trial_id')}")
print(f"Last frame timestamp: {last_f.get('timestamp')}, frame_idx: {last_f.get('frame_idx')}, trial_id: {last_f.get('trial_id')}")

# 5. Model status
sh_statuses = collections.Counter(r.get("static_head", {}).get("load_status") for _, r in lines)
has_sh = collections.Counter(r.get("has_static_head") for _, r in lines)
print(f"\nStatic Head load_status: {dict(sh_statuses)}")
print(f"has_static_head: {dict(has_sh)}")

# 6. Prediction sources
sources = collections.Counter(r.get("prediction_source") for _, r in lines)
print(f"Prediction sources: {dict(sources)}")

# 7. Confirmed flags & new tokens
confirmed_cnt = collections.Counter(r.get("confirmed") for _, r in lines)
print(f"Confirmed distribution: {dict(confirmed_cnt)}")

# 8. Check timestamps ordering per trial
for t_id in sorted(trials.keys()):
    t_frames = [r for _, r in lines if r.get("trial_id") == t_id]
    ts_list = [f.get("timestamp") for f in t_frames if f.get("timestamp") is not None]
    is_sorted = all(ts_list[i] <= ts_list[i+1] for i in range(len(ts_list)-1))
    f_indices = [f.get("frame_idx") for f in t_frames if f.get("frame_idx") is not None]
    is_f_idx_sorted = all(f_indices[i] <= f_indices[i+1] for i in range(len(f_indices)-1))
    duration = (ts_list[-1] - ts_list[0]) if len(ts_list) > 1 else 0.0
    print(f"Trial {t_id}: {len(t_frames)} frames, duration={duration:.2f}s, ts_monotonic={is_sorted}, f_idx_monotonic={is_f_idx_sorted}")

