import json

file_path = "backend/data/diagnostics/pilot_pilot_20261108_0321.jsonl"

with open(file_path, "r", encoding="utf-8") as f:
    for line_idx, line in enumerate(f, start=1):
        d = json.loads(line)
        if d.get("trial_id") == "pilot_20261108_0321_H_rep01_att01" and d.get("final_sign") == "H":
            print("=" * 60)
            print("SAMPLE H PREDICTED AS H (Line", line_idx, ")")
            print("=" * 60)
            print(json.dumps(d, indent=2))
            break

with open(file_path, "r", encoding="utf-8") as f:
    for line_idx, line in enumerate(f, start=1):
        d = json.loads(line)
        if d.get("trial_id") == "pilot_20261108_0321_H_rep03_att01" and d.get("final_sign") == "R":
            print("=" * 60)
            print("SAMPLE H PREDICTED AS R (Line", line_idx, ")")
            print("=" * 60)
            print(json.dumps(d, indent=2))
            break

with open(file_path, "r", encoding="utf-8") as f:
    for line_idx, line in enumerate(f, start=1):
        d = json.loads(line)
        if d.get("trial_id") == "pilot_20261108_0321_H_rep02_att01" and d.get("final_sign") == "NEUTRAL":
            print("=" * 60)
            print("SAMPLE H PREDICTED AS NEUTRAL (Line", line_idx, ")")
            print("=" * 60)
            print(json.dumps(d, indent=2))
            break

