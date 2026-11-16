import json

filepath = "backend/data/diagnostics/pilot_pilot_20261108_193712.jsonl"
with open(filepath, "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip()]

confirmed_records = []
for i, r in enumerate(records):
    if r.get("confirmed"):
        confirmed_records.append((i, r))

print(f"Total confirmed records: {len(confirmed_records)}")

for idx in range(len(confirmed_records) - 1):
    l1, r1 = confirmed_records[idx]
    l2, r2 = confirmed_records[idx + 1]
    t1 = r1["timestamp"]
    t2 = r2["timestamp"]
    dt = t2 - t1
    if dt < 1.21:
        print(f"Pair {idx} -> {idx+1}: Line {l1} (trial={r1.get('trial_id')}, frame={r1.get('frame_idx')}, t1={t1:.4f}) to Line {l2} (trial={r2.get('trial_id')}, frame={r2.get('frame_idx')}, t2={t2:.4f}) -> dt = {dt:.6f}")

