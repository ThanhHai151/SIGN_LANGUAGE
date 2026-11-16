import re

with open("backend/vision/recognizer.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

for idx, line in enumerate(lines):
    if "last_confirmed_time" in line or "cooldown_sec" in line:
        print(f"Line {idx+1}: {line.strip()}")

