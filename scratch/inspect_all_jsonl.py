import glob
import json
import os

files = glob.glob("**/*.jsonl", recursive=True)
print(f"Inspecting {len(files)} JSONL files:\n")

for fpath in files:
    size = os.path.getsize(fpath)
    print(f"=== {fpath} ({size} bytes) ===")
    sample_keys = set()
    has_landmarks = False
    expected_signs = set()
    sessions = set()
    trials = set()
    count = 0
    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            count += 1
            try:
                data = json.loads(line)
                sample_keys.update(data.keys())
                if "landmarks" in data or "raw_landmarks" in data or "hand_landmarks" in data or "coords" in data:
                    has_landmarks = True
                if "expected_sign" in data:
                    expected_signs.add(data["expected_sign"])
                if "session_id" in data:
                    sessions.add(data["session_id"])
                if "trial_id" in data:
                    trials.add(data["trial_id"])
            except Exception as e:
                print(f"  Parse error on line {count}: {e}")
    print(f"  Total lines: {count}")
    print(f"  Keys present: {sorted(list(sample_keys))}")
    print(f"  Contains landmarks: {has_landmarks}")
    print(f"  Expected signs: {expected_signs}")
    print(f"  Sessions: {sessions}")
    print(f"  Trials count: {len(trials)}")
    print()

