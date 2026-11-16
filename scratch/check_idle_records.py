import json
from collections import Counter

filepath = "backend/data/diagnostics/pilot_pilot_20261108_193712.jsonl"
with open(filepath, "r", encoding="utf-8") as f:
    lines = [json.loads(l) for l in f if l.strip()]

idle = [r for r in lines if r.get("trial_id") is None]
print(f"Total idle records: {len(idle)}")
for i, r in enumerate(idle):
    print(f"Idle {i}: sess={r.get('session_id')}, frame_idx={r.get('frame_idx')}, expected={r.get('expected_sign')}, final={r.get('final_sign')}, geom_status={r.get('geometric',{}).get('status')}")

