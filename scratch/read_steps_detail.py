import json

transcript_path = r"C:\Users\PC\.gemini\antigravity\brain\e7c77c34-c401-4b3d-8757-568dc34b3e88\.system_generated\logs\transcript_full.jsonl"

target_steps = [4671, 4672, 4673, 4716, 4717, 4718]

with open(transcript_path, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        st = obj.get("step_index")
        if st in target_steps:
            print(f"==================================================")
            print(f"STEP {st} | TYPE: {obj.get('type')} | CREATED_AT: {obj.get('created_at')}")
            print(f"SOURCE: {obj.get('source')} | STATUS: {obj.get('status')}")
            tc = obj.get("tool_calls")
            if tc:
                print("TOOL_CALLS:", json.dumps(tc, indent=2))
            cnt = obj.get("content")
            if cnt:
                print("CONTENT:")
                print(cnt)
            print()

