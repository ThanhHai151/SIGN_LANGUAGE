import json

with open(r'C:\Users\PC\.gemini\antigravity\brain\e7c77c34-c401-4b3d-8757-568dc34b3e88\.system_generated\logs\transcript_full.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        data = json.loads(line)
        c = data.get('content', '')
        if 'Showing lines' in c and 'alphabet_classifier' in c:
            lines = c.splitlines()
            hdr = lines[2] if len(lines) > 2 else ''
            print(f"Line {i}: {hdr}")

