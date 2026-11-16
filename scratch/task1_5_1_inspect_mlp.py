import sys
import os
import json
import collections

filepath = "backend/data/diagnostics/pilot_pilot_20261108_183703.jsonl"

with open(filepath, "r", encoding="utf-8") as f:
    lines = [json.loads(line) for line in f if line.strip()]

sh_data = [r.get("static_head", {}) for r in lines]

load_statuses = collections.Counter(s.get("load_status") for s in sh_data)
has_static_head_vals = collections.Counter(r.get("has_static_head") for r in lines)
raw_letters = collections.Counter(s.get("raw_predicted_letter") for s in sh_data if s.get("raw_predicted_letter"))
confidences = [s.get("confidence") for s in sh_data if s.get("confidence") is not None]

print("=== STATIC HEAD RUNTIME STATUS ===")
print(f"load_status: {dict(load_statuses)}")
print(f"has_static_head: {dict(has_static_head_vals)}")
print(f"Raw letters predicted: {dict(raw_letters)}")
if confidences:
    print(f"Confidences: min={min(confidences)}, max={max(confidences)}, mean={sum(confidences)/len(confidences):.4f}")
else:
    print("No raw MLP predictions recorded (because geometric matched P directly or MLP was not called).")

