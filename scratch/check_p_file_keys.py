import json

p_file = "backend/data/diagnostics/pilot_pilot_20261108_183703.jsonl"
with open(p_file, "r") as f:
    for line in f:
        d = json.loads(line)
        if d.get("trial_id"):
            print("P file top keys:", list(d.keys()))
            print("P file 'landmarks' present?", "landmarks" in d)
            print("P file 'geometric' features present?", "features" in d.get("geometric", {}))
            break

