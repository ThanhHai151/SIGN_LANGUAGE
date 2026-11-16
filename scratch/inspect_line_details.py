import json

latest = 'backend/data/diagnostics/pilot_pilot_20261106_2347.jsonl'
with open(latest, 'r', encoding='utf-8') as f:
    lines = [line.strip() for line in f if line.strip()]

for i in range(1, 10):
    data = json.loads(lines[i])
    print(f"=== LINE {i+1} ===")
    print("Top-level keys:", list(data.keys()))
    for k, v in data.items():
        if k != "landmarks":
            print(f"  {k}: {v}")
    if "landmarks" in data:
        lms = data["landmarks"]
        print(f"  landmarks type={type(lms)}, len={len(lms) if isinstance(lms, list) else None}")

