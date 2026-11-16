import json
from collections import Counter

file_path = "backend/data/diagnostics/pilot_pilot_20261108_0321.jsonl"

with open(file_path, "r", encoding="utf-8") as f:
    lines = [json.loads(line) for line in f]

# Lines 12 to 73 is index 11 to 72
block_12_73 = lines[11:73]
signs = Counter(d.get("final_sign") for d in block_12_73)
sources = Counter(d.get("prediction_source") for d in block_12_73)
print("Block Lines 12..73:")
print("  Signs:", dict(signs))
print("  Sources:", dict(sources))

