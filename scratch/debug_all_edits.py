import json
import subprocess

p = r'C:\Users\PC\.gemini\antigravity\brain\e7c77c34-c401-4b3d-8757-568dc34b3e88\.system_generated\logs\transcript_full.jsonl'

edit_steps = [323, 327, 331, 335, 343, 347, 351, 357, 560, 562, 566, 574, 576, 588, 594, 622, 631, 635, 679, 683, 1055, 1059, 1063, 1067, 1071, 1077, 1083, 1113]

edits = {}
with open(p, 'r', encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        idx = d.get('step_index')
        if idx in edit_steps:
            calls = d.get('tool_calls', [])
            for c in calls:
                args = c.get('args', {})
                if 'alphabet_classifier.py' in args.get('Path', ''):
                    edits[idx] = (args.get('Target'), args.get('Replacement'))

print(f"Total edits recorded: {len(edits)}")

out = subprocess.check_output(['git', 'cat-file', '-p', '67357e36:backend/vision/alphabet_classifier.py'])
text = out.decode('utf-8')

for idx in sorted(edits.keys()):
    target, replacement = edits[idx]
    if target in text:
        text = text.replace(target, replacement, 1)
        print(f"Applied {idx} successfully")
    else:
        print(f"MISSING {idx}")
        print("Target:", repr(target[:60]))


