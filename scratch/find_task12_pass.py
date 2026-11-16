import json

with open("C:/Users/PC/.gemini/antigravity/brain/ade30714-1637-462e-8636-6df333d2e57d/.system_generated/logs/transcript.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        content = data.get("content", "")
        if "TASK 1.2 OFFLINE PASS" in content:
            print(f"Step {data.get('step_index')}")
            for l in content.split("\n"):
                if "198" in l or "regression" in l.lower() or "test" in l.lower() or "suite" in l.lower():
                    print(f"  {l}")

