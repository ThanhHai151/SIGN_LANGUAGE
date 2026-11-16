import json

filepath = "backend/data/diagnostics/pilot_pilot_20261108_193712.jsonl"
with open(filepath, "r", encoding="utf-8") as f:
    lines = [json.loads(l) for l in f if l.strip()]

for idx in [108, 109, 110, 111, 112, 113, 114, 115]:
    r = lines[idx]
    print(f"Line {idx}: sess={r.get('session_id')}, trial={r.get('trial_id')}, frame={r.get('frame_idx')}, sign={r.get('final_sign')}, conf={r.get('final_confidence')}, confirmed={r.get('confirmed')}")

