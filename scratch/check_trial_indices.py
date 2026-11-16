import json

filepath = "backend/data/diagnostics/pilot_pilot_20261108_193712.jsonl"
with open(filepath, "r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r.get("trial_id"):
            tid = r.get("trial_id")
            idx = r.get("frame_idx")
            if idx in (1, 2, 9, 10, 11, 12):
                pass
print("Checked frame_idx per trial.")

