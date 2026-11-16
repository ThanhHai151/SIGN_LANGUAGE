import json

with open("C:/Users/PC/.gemini/antigravity/brain/ade30714-1637-462e-8636-6df333d2e57d/.system_generated/logs/transcript.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        idx = data.get("step_index")
        if idx and 1680 <= idx <= 1703:
            tc = data.get("tool_calls")
            if tc:
                print(f"Step {idx}: {tc}")

