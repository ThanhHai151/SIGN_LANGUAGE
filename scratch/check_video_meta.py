import json

with open("backend/data/alphabet_v2/metadata/videos_metadata.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

for k in ["P", "Q", "R", "X"]:
    if k in meta:
        print(f"Letter {k}: {meta[k]}")
    else:
        # Check by item
        matches = [item for item in meta if (isinstance(item, dict) and item.get("letter") == k) or k in str(item)]
        print(f"Letter {k} in list: {matches[:2]}")
