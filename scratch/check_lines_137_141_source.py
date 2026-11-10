import json

filepath = "backend/data/diagnostics/pilot_pilot_20261108_193712.jsonl"
with open(filepath, "r", encoding="utf-8") as f:
    lines = [json.loads(l) for l in f if l.strip()]

for idx in [135, 136, 137, 138, 139, 140, 141]:
    r = lines[idx]
    print(f"Line {idx}:")
    print(f"  sign: {r.get('final_sign')}")
    print(f"  confirmed: {r.get('confirmed')}")
    print(f"  prediction_source: {r.get('prediction_source')}")
    print(f"  reject_reason: {r.get('reject_reason')}")
    print(f"  gating_status: {r.get('gating_status')}")
    print(f"  geom: status={r.get('geometric',{}).get('status')}, letter={r.get('geometric',{}).get('letter')}")

