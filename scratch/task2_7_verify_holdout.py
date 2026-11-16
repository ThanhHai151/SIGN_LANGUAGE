"""
TASK 2.7: Independent Webcam Holdout Verification for Letter X
File: backend/data/diagnostics/pilot_pilot_20261108_193712.jsonl
"""

import sys
import os
import json
import hashlib
from collections import defaultdict, Counter
import numpy as np

FILE_PATH = "backend/data/diagnostics/pilot_pilot_20261108_193712.jsonl"

def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=" * 70)
    print("TASK 2.7: INDEPENDENT WEBCAM HOLDOUT VERIFICATION FOR LETTER X")
    print("=" * 70)
    
    if not os.path.exists(FILE_PATH):
        print(f"ERROR: File not found: {FILE_PATH}")
        sys.exit(1)
        
    file_size = os.path.getsize(FILE_PATH)
    file_hash = sha256_file(FILE_PATH)
    print(f"File Path: {FILE_PATH}")
    print(f"File Size: {file_size:,} bytes")
    print(f"SHA256: {file_hash}")
    
    total_lines = 0
    parse_errors = 0
    records = []
    
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            total_lines += 1
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append(rec)
            except Exception as e:
                parse_errors += 1
                print(f"Parse error at line {idx+1}: {e}")
                
    print(f"\n1. INTEGRITY CHECK:")
    print(f"- Total lines in file: {total_lines}")
    print(f"- Valid JSON records: {len(records)}")
    print(f"- Parse errors: {parse_errors}")
    
    # Session info
    sessions = set(r.get("session_id") for r in records)
    print(f"- Session IDs found: {sessions}")
    
    expected_signs = set(r.get("expected_sign") for r in records)
    print(f"- Expected signs found: {expected_signs}")
    
    # Check monotonicity of frame_idx and timestamps
    frame_indices = [r.get("frame_idx") for r in records if r.get("frame_idx") is not None]
    timestamps = [r.get("timestamp") for r in records if r.get("timestamp") is not None]
    
    frame_idx_sorted = all(frame_indices[i] <= frame_indices[i+1] for i in range(len(frame_indices)-1))
    ts_sorted = all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1))
    print(f"- Frame index monotonic: {frame_idx_sorted} (min: {min(frame_indices) if frame_indices else None}, max: {max(frame_indices) if frame_indices else None})")
    print(f"- Timestamp monotonic: {ts_sorted} (duration: {timestamps[-1] - timestamps[0]:.2f}s)" if timestamps else "No timestamps")
    
    # Check landmarks
    landmarks_valid = 0
    landmarks_missing = 0
    for r in records:
        lms = r.get("landmarks")
        if lms and isinstance(lms, list) and len(lms) == 21:
            landmarks_valid += 1
        else:
            landmarks_missing += 1
    print(f"- Landmarks valid (21 points): {landmarks_valid}")
    print(f"- Landmarks missing/invalid: {landmarks_missing}")
    
    # Check static head availability
    static_heads = Counter(r.get("has_static_head") for r in records)
    print(f"- Has static head distribution: {static_heads}")
    load_statuses = Counter(r.get("static_head", {}).get("load_status") for r in records)
    print(f"- Static head load status: {load_statuses}")
    
    # Partition records into Active vs Idle
    active_records = [r for r in records if r.get("trial_id") is not None and r.get("expected_sign") is not None]
    idle_records = [r for r in records if r.get("trial_id") is None or r.get("expected_sign") is None]
    print(f"\n- Active trial records: {len(active_records)}")
    print(f"- Idle records: {len(idle_records)}")
    
    # Group active records by trial_id
    trials = defaultdict(list)
    for r in active_records:
        trials[r.get("trial_id")].append(r)
        
    print(f"\nTotal Trials Found: {len(trials)}")
    for tid, frames in sorted(trials.items()):
        first_ts = frames[0].get("timestamp")
        last_ts = frames[-1].get("timestamp")
        dur = last_ts - first_ts if first_ts and last_ts else 0.0
        preds = Counter(f.get("final_sign") for f in frames)
        tokens = [f.get("new_token") for f in frames if f.get("new_token")]
        conf_frames = [f for f in frames if f.get("confirmed")]
        print(f"  {tid}: {len(frames)} frames, duration={dur:.2f}s, preds={dict(preds)}, confirmed_frames={len(conf_frames)}, tokens={tokens}")

if __name__ == "__main__":
    main()

