import json
import subprocess

out = subprocess.check_output(['git', 'cat-file', '-p', '67357e36:backend/vision/alphabet_classifier.py'])
text = out.decode('utf-8')

p = r'C:\Users\PC\.gemini\antigravity\brain\e7c77c34-c401-4b3d-8757-568dc34b3e88\.system_generated\logs\transcript_full.jsonl'

edit_steps = [1055, 1059, 1063, 1067, 1071, 1077, 1083, 1113]
edits = []

with open(p, 'r', encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        idx = d.get('step_index')
        if idx in edit_steps:
            calls = d.get('tool_calls', [])
            for c in calls:
                args = c.get('args', {})
                target = args.get('Target')
                replacement = args.get('Replacement')
                edits.append((idx, target, replacement))

print(f"Loaded {len(edits)} edits for alphabet_classifier.py")

for idx, target, replacement in edits:
    print(f"Applying step {idx}...")
    if target not in text:
        print(f"  FAILED to find target for step {idx}!")
        print("  Target preview:", repr(target[:80]))
    else:
        text = text.replace(target, replacement, 1)
        print(f"  Successfully applied step {idx}")

with open('backend/vision/alphabet_classifier.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Saved restored alphabet_classifier.py!")


