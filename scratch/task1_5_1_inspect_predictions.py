import sys
import os
import json
import collections

filepath = "backend/data/diagnostics/pilot_pilot_20261108_183703.jsonl"

with open(filepath, "r", encoding="utf-8") as f:
    lines = [json.loads(line) for line in f if line.strip()]

# Filter active trial frames (expected_sign == 'P')
active_frames = [r for r in lines if r.get("expected_sign") == "P"]
idle_frames = [r for r in lines if r.get("expected_sign") is None]

print(f"Total lines: {len(lines)}")
print(f"Active trial frames (expected_sign=='P'): {len(active_frames)}")
print(f"Idle/inter-trial frames (expected_sign==None): {len(idle_frames)}")

# Predictions breakdown in active frames
geom_status_cnt = collections.Counter(r.get("geometric", {}).get("status") for r in active_frames)
geom_letter_cnt = collections.Counter(r.get("geometric", {}).get("letter") for r in active_frames)
final_sign_cnt = collections.Counter(r.get("final_sign") for r in active_frames)
source_cnt = collections.Counter(r.get("prediction_source") for r in active_frames)
confirmed_cnt = collections.Counter(r.get("confirmed") for r in active_frames)
confirmed_by_sign = collections.Counter(r.get("final_sign") for r in active_frames if r.get("confirmed"))

print("\n=== ACTIVE TRIAL FRAMES SUMMARY ===")
print(f"Geometric status: {dict(geom_status_cnt)}")
print(f"Geometric letters: {dict(geom_letter_cnt)}")
print(f"Final signs: {dict(final_sign_cnt)}")
print(f"Prediction sources: {dict(source_cnt)}")
print(f"Confirmed distribution: {dict(confirmed_cnt)}")
print(f"Confirmed signs: {dict(confirmed_by_sign)}")

# Check what the 2 non-P final signs are!
non_p_neutral = [r for r in active_frames if r.get("final_sign") not in ("P", "NEUTRAL")]
print(f"\nNon-P and Non-Neutral frames in active trials: {len(non_p_neutral)}")
for r in non_p_neutral:
    print(f"  Trial: {r.get('trial_id')}, FrameIdx: {r.get('frame_idx')}, Time: {r.get('timestamp')}, Sign: {r.get('final_sign')}, Conf: {r.get('final_confidence')}, Source: {r.get('prediction_source')}, Confirmed: {r.get('confirmed')}, GeomStatus: {r.get('geometric', {}).get('status')}")

# Check predictions in IDLE frames
print("\n=== IDLE / INTER-TRIAL FRAMES SUMMARY ===")
idle_final_signs = collections.Counter(r.get("final_sign") for r in idle_frames)
idle_geom_letters = collections.Counter(r.get("geometric", {}).get("letter") for r in idle_frames)
print(f"Idle final signs: {dict(idle_final_signs)}")
print(f"Idle geom letters: {dict(idle_geom_letters)}")

