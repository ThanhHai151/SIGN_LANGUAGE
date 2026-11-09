import json

filepath = "backend/data/diagnostics/pilot_pilot_20261108_193712.jsonl"
with open(filepath, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        data = json.loads(line)
        if data.get("trial_id"):
            geom = data.get("geometric", {})
            print("geom keys:", list(geom.keys()))
            for k, v in geom.items():
                if k != "features":
                    print(f"  {k}: {v}")
                else:
                    print(f"  features count: {len(v)}")
            break

