import json

with open("C:/Users/PC/.gemini/antigravity/brain/ade30714-1637-462e-8636-6df333d2e57d/.system_generated/logs/transcript.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        if data.get("step_index") == 1704:
            print(data.get("content"))

