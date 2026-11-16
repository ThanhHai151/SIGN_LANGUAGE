import json

transcript_path = r"C:\Users\PC\.gemini\antigravity\brain\e7c77c34-c401-4b3d-8757-568dc34b3e88\.system_generated\logs\transcript_full.jsonl"

for target in [4671, 4716]:
    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("step_index") == target:
                print(f"==================== STEP {target} ====================")
                print("SOURCE:", obj.get("source"))
                print("TYPE:", obj.get("type"))
                print("TOOL_CALLS:", [tc.get("name") for tc in obj.get("tool_calls", [])])
                cnt = obj.get("content", "")
                print(f"CONTENT ({len(cnt)} chars):")
                print(cnt[:600])
                print("...")
                print(cnt[-400:])
                print()

