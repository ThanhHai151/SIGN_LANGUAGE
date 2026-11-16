import json

with open("C:/Users/PC/.gemini/antigravity/brain/ade30714-1637-462e-8636-6df333d2e57d/.system_generated/logs/transcript.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        if data.get("step_index") in [1981, 1985, 1988, 1989, 1991, 1992, 1993, 1994, 1995]:
            print(f"Step {data.get('step_index')}: {data.get('tool_calls') or data.get('content')[:150]}")

