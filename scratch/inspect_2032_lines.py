import json

with open("backend/data/diagnostics/pilot_pilot_20261105_2032.jsonl", "r", encoding="utf-8") as f:
    lines = [json.loads(line) for line in f if line.strip()]

print(f"Total lines in pilot_20261105_2032.jsonl: {len(lines)}")
for i, d in enumerate(lines):
    ts = d.get("timestamp")
    tid = d.get("trial_id")
    fidx = d.get("frame_idx")
    exp = d.get("expected_sign")
    fs = d.get("final_sign")
    conf = d.get("final_confidence")
    src = d.get("prediction_source")
    geom_l = d.get("geometric", {}).get("letter")
    geom_s = d.get("geometric", {}).get("status")
    print(f"line {i:03d} | ts={ts} | trial={tid} | frame={fidx} | exp={exp} | final={fs:7s} ({conf:.2f}) | src={src:14s} | geom={geom_s}:{geom_l}")

