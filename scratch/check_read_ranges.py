import json

p = r'C:\Users\PC\.gemini\antigravity\brain\e7c77c34-c401-4b3d-8757-568dc34b3e88\.system_generated\logs\transcript_full.jsonl'

steps = [1111, 1256, 1320, 1330, 1332, 1370, 1372, 1374]

with open(p, 'r', encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        idx = d.get('step_index')
        if idx in steps:
            content = d.get('content', '')
            lines = content.split('\n')
            header = [l for l in lines if 'Showing lines' in l]
            print(f"Step {idx}: {header}")


