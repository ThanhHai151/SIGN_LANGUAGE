import json

with open("C:/Users/PC/.gemini/antigravity/brain/ade30714-1637-462e-8636-6df333d2e57d/.system_generated/logs/transcript.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        for tc in data.get("tool_calls", []):
            cmd = tc.get("args", {}).get("CommandLine", "")
            if "pytest" in cmd:
                print(f"Step {data.get('step_index')}: {cmd}")

