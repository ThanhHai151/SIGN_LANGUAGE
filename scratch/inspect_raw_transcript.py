import json

transcript_path = r"C:\Users\PC\.gemini\antigravity\brain\e7c77c34-c401-4b3d-8757-568dc34b3e88\.system_generated\logs\transcript.jsonl"

steps_to_inspect = [4671, 4672, 4673, 4716, 4717, 4718, 4749, 4750, 4751]

with open(transcript_path, "r", encoding="utf-8") as f:
    for line_idx, line in enumerate(f):
        obj = json.loads(line)
        if obj.get("step_index") in steps_to_inspect:
            print(f"=== Line {line_idx+1}: Step {obj.get('step_index')} | Type: {obj.get('type')} | Source: {obj.get('source')} ===")
            print("Keys:", list(obj.keys()))
            if "content" in obj:
                cnt = str(obj["content"])
                print(f"Content length: {len(cnt)}")
                print("Content first 200 chars:")
                print(repr(cnt[:200]))
            if "thinking" in obj:
                th = str(obj["thinking"])
                print(f"Thinking length: {len(th)}")
                print("Thinking first 200 chars:")
                print(repr(th[:200]))
            print()

