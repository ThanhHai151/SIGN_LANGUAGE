import json

with open("backend/data/diagnostics/pilot_pilot_20261105_2024.jsonl", "r", encoding="utf-8") as f:
    lines = [json.loads(line) for line in f if line.strip()]

print(f"Total lines in pilot_20261105_2024.jsonl: {len(lines)}")
for i, d in enumerate(lines):
    ts = d.get("timestamp")
    tid = d.get("trial_id")
    fidx = d.get("frame_idx")
    exp = d.get("expected_sign")
    fs = d.get("final_sign")
    conf = d.get("final_confidence")
    src = d.get("prediction_source")
    geom = d.get("geometric", {})
    gl = geom.get("letter")
    gs = geom.get("status")
    print(f"l{i:03d} | ts={ts} | trial={tid} | frame={fidx} | exp={exp} | final={fs:7s} ({conf:.2f}) | src={src:14s} | geom={gs}:{gl}")

