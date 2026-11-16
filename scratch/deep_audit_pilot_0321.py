import json
import hashlib
import os
from collections import Counter, defaultdict

file_path = "backend/data/diagnostics/pilot_pilot_20261108_0321.jsonl"

# Compute SHA-256
h = hashlib.sha256()
with open(file_path, "rb") as f:
    while chunk := f.read(65536):
        h.update(chunk)
sha256 = h.hexdigest()
file_size = os.path.getsize(file_path)

lines = []
with open(file_path, "r", encoding="utf-8") as f:
    for line_idx, raw_line in enumerate(f, start=1):
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        data = json.loads(raw_line)
        lines.append((line_idx, data))

total_lines = len(lines)
idle_lines = [item for item in lines if item[1].get("session_id") is None]
trial_lines = [item for item in lines if item[1].get("session_id") is not None]

print("=" * 70)
print("AUDIT FILE INFORMATION")
print("=" * 70)
print(f"File Path: {file_path}")
print(f"File Size: {file_size:,} bytes")
print(f"SHA-256: {sha256}")
print(f"Total Lines: {total_lines}")
print(f"Idle Stream Lines (session_id=None): {len(idle_lines)}")
print(f"Controlled Trial Lines: {len(trial_lines)}")

# Check sessions in trial_lines
trial_sessions = Counter(item[1].get("session_id") for item in trial_lines)
print(f"Trial Sessions: {dict(trial_sessions)}")

# Group trial lines by trial_id
trials_dict = defaultdict(list)
for line_idx, d in trial_lines:
    tr_id = d.get("trial_id")
    trials_dict[tr_id].append((line_idx, d))

print(f"\nTotal Controlled Trials: {len(trials_dict)}")
for tr_id, fr_list in sorted(trials_dict.items()):
    exp_sign = fr_list[0][1].get("expected_sign")
    frame_indices = [d.get("frame_idx") for _, d in fr_list]
    min_idx, max_idx = min(frame_indices), max(frame_indices)
    expected_range = list(range(1, len(frame_indices) + 1))
    is_sequential = (frame_indices == expected_range)
    timestamps = [d.get("timestamp") for _, d in fr_list if d.get("timestamp")]
    duration = (max(timestamps) - min(timestamps)) if len(timestamps) > 1 else 0
    
    # Counts of final_sign
    sign_counts = Counter(d.get("final_sign") for _, d in fr_list)
    src_counts = Counter(d.get("prediction_source") for _, d in fr_list)
    
    print(f"Trial: {tr_id}")
    print(f"  Expected: {exp_sign} | Frames: {len(fr_list)} | frame_idx: [{min_idx}..{max_idx}] | Sequential 1..N: {is_sequential}")
    print(f"  Duration: {duration:.2f}s | Signs: {dict(sign_counts)} | Sources: {dict(src_counts)}")

print("\n" + "=" * 70)
print("IDLE LINES ANALYSIS")
print("=" * 70)
idle_sign_counts = Counter(d.get("final_sign") for _, d in idle_lines)
idle_src_counts = Counter(d.get("prediction_source") for _, d in idle_lines)
print(f"Idle signs: {dict(idle_sign_counts)}")
print(f"Idle sources: {dict(idle_src_counts)}")
line_ranges = []
# Find contiguous blocks of idle lines
curr_block = []
for line_idx, d in lines:
    is_idle = (d.get("session_id") is None)
    if is_idle:
        curr_block.append(line_idx)
    else:
        if curr_block:
            line_ranges.append((curr_block[0], curr_block[-1], len(curr_block), "idle"))
            curr_block = []
if curr_block:
    line_ranges.append((curr_block[0], curr_block[-1], len(curr_block), "idle"))

print("\nDistribution of lines in file (order of blocks):")
curr_trial = None
curr_trial_block = []
block_summary = []
for line_idx, d in lines:
    tr = d.get("trial_id")
    if tr != curr_trial:
        if curr_trial_block:
            block_summary.append((curr_trial_block[0], curr_trial_block[-1], len(curr_trial_block), curr_trial))
        curr_trial = tr
        curr_trial_block = [line_idx]
    else:
        curr_trial_block.append(line_idx)
if curr_trial_block:
    block_summary.append((curr_trial_block[0], curr_trial_block[-1], len(curr_trial_block), curr_trial))

for start_l, end_l, cnt, label in block_summary:
    print(f"  Lines {start_l:3d}..{end_l:3d} ({cnt:2d} frames): {label if label else '[IDLE STREAM BACKGROUND]'}")

