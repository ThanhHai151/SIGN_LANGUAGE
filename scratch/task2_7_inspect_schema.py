import json

filepath = "backend/data/diagnostics/pilot_pilot_20261108_193712.jsonl"
with open(filepath, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        data = json.loads(line)
        if data.get("trial_id"):
            print(f"Record {i} keys:", list(data.keys()))
            print("trial_id:", data.get("trial_id"))
            print("frame_idx:", data.get("frame_idx"))
            print("expected_sign:", data.get("expected_sign"))
            print("final_sign:", data.get("final_sign"))
            print("confirmed:", data.get("confirmed"))
            print("prediction_source:", data.get("prediction_source"))
            print("geometric status:", data.get("geometric", {}).get("status"))
            print("geometric letter:", data.get("geometric", {}).get("letter"))
            print("geometric rule_name:", data.get("geometric", {}).get("rule_name"))
            print("geometric features:", list(data.get("geometric", {}).get("features", {}).keys()))
            if "landmarks" in data:
                print("landmarks type/len:", type(data["landmarks"]), len(data["landmarks"]) if data["landmarks"] else None)
            break

