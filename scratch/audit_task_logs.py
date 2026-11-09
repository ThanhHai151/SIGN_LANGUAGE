import json
import os

transcript_path = r"C:\Users\PC\.gemini\antigravity\brain\e7c77c34-c401-4b3d-8757-568dc34b3e88\.system_generated\logs\transcript.jsonl"

matching_lines = []
with open(transcript_path, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        if "task-4664" in line or "task-4713" in line:
            obj = json.loads(line)
            matching_lines.append((idx + 1, obj))

print(f"Total matching lines in transcript.jsonl: {len(matching_lines)}")
for line_num, obj in matching_lines:
    print(f"\n--- Line {line_num} (step {obj.get('step_index')}, type={obj.get('type')}) ---")
    tool_calls = obj.get("tool_calls", [])
    if tool_calls:
        for tc in tool_calls:
            print("Tool call:", tc.get("name"), tc.get("arguments"))
    content = obj.get("content")
    if content:
        print("Content:", str(content)[:300])

