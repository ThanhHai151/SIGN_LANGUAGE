import json

for fname in [
    "backend/data/diagnostics/pilot_eval_29_pilot_20261111_172142_27052b_20261111_172433_frames.jsonl",
    "backend/data/diagnostics/pilot_eval_29_pilot_20261112_040819_b6dc69_20261112_041644_frames.jsonl",
    "backend/data/diagnostics/pilot_eval_29_pilot_20261112_041651_86b712_20261112_041820_frames.jsonl"
]:
    with open(fname, "r", encoding="utf-8") as f:
        not_none = sum(1 for line in f if json.loads(line).get("landmarks") is not None)
    print(f"{fname}: {not_none} frames have landmarks != None")

