import sys
sys.path.insert(0, ".")
import numpy as np
import json
import glob

# Load static landmarks json files
all_files = glob.glob("backend/data/alphabet_v2/static_landmarks/*.json")
print(f"Total static landmark files: {len(all_files)}")

dataset = {}
for f in all_files:
    data = json.load(open(f, encoding="utf-8"))
    let = data.get("letter")
    samples = [s for s in data.get("samples", []) if s.get("phase") == "stable_pose" and s.get("landmarks") is not None]
    dataset[let] = samples
    print(f"Letter {let}: {len(samples)} stable samples")

