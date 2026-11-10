import json

filepath = "backend/data/diagnostics/pilot_pilot_20261108_193712.jsonl"
with open(filepath, "r", encoding="utf-8") as f:
    lines = [json.loads(l) for l in f if l.strip()]

for idx in range(110, 116):
    r = lines[idx]
    print(f"Line {idx}:")
    print(f"  timestamp: {r.get('timestamp')}")
    print(f"  trial_id: {r.get('trial_id')}")
    print(f"  frame_idx: {r.get('frame_idx')}")
    print(f"  final_sign: {r.get('final_sign')}")
    print(f"  confirmed: {r.get('confirmed')}")
    print(f"  prediction_source: {r.get('prediction_source')}")
    print(f"  geometric.status: {r.get('geometric', {}).get('status')}")

