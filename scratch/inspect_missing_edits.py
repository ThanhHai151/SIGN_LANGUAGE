import json

p = r'C:\Users\PC\.gemini\antigravity\brain\e7c77c34-c401-4b3d-8757-568dc34b3e88\.system_generated\logs\transcript_full.jsonl'
check_steps = [560, 562, 574, 576, 1071]

with open(p, 'r', encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        idx = d.get('step_index')
        if idx in check_steps:
            calls = d.get('tool_calls', [])
            for c in calls:
                args = c.get('args', {})
                print(f"=== STEP {idx} ===")
                print("Target:\n", args.get('Target'))
                print("Replacement:\n", args.get('Replacement'))


