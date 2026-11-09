import json

transcript_path = r"C:\Users\PC\.gemini\antigravity\brain\e7c77c34-c401-4b3d-8757-568dc34b3e88\.system_generated\logs\transcript_full.jsonl"

with open(transcript_path, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        st = obj.get("step_index")
        if st in [4663, 4664, 4712, 4713]:
            print(f"=== Step {st} ({obj.get('type')}) ===")
            tc = obj.get("tool_calls")
            if tc:
                print("tool_calls:", json.dumps(tc, indent=2))
            cnt = obj.get("content")
            if cnt:
                print("content:", cnt[:300])
            print()

