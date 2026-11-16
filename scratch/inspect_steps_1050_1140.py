import json

p = r'C:\Users\PC\.gemini\antigravity\brain\e7c77c34-c401-4b3d-8757-568dc34b3e88\.system_generated\logs\transcript_full.jsonl'

with open(p, 'r', encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        idx = d.get('step_index')
        if 1050 <= idx <= 1140:
            calls = d.get('tool_calls', [])
            for c in calls:
                print(f"Step {idx}: {c.get('name')} {c.get('args', {}).get('Path') or c.get('args', {}).get('CommandLine', '')[:50]}")


