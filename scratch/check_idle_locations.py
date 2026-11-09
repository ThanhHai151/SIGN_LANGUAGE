import json

filepath = "backend/data/diagnostics/pilot_pilot_20261108_193712.jsonl"
with open(filepath, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        r = json.loads(line)
        if r.get("trial_id") is None:
            print(f"Line {i}: idle, ts={r.get('timestamp')}")
        elif i == 0 or (i > 0 and prev_trial is None):
            print(f"Line {i}: START TRIAL {r.get('trial_id')}, ts={r.get('timestamp')}")
        prev_trial = r.get("trial_id")

