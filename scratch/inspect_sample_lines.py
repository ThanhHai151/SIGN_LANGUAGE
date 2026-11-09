import json

file_path = "backend/data/diagnostics/pilot_pilot_20261108_0321.jsonl"

with open(file_path, "r", encoding="utf-8") as f:
    for i in range(10):
        line = f.readline()
        if not line:
            break
        data = json.loads(line)
        print(f"--- Line {i+1} keys: {list(data.keys())} ---")
        # print essential fields
        sample = {k: data[k] for k in ["session_id", "trial_id", "frame_idx", "expected_sign", "final_sign", "sign", "prediction", "prediction_source", "gating_status"] if k in data}
        print(f"Sample: {sample}")

