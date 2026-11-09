import json
import subprocess

# Start from git commit 67357e36
out = subprocess.check_output(['git', 'cat-file', '-p', '67357e36:backend/vision/alphabet_classifier.py'])
text = out.decode('utf-8')

p = r'C:\Users\PC\.gemini\antigravity\brain\e7c77c34-c401-4b3d-8757-568dc34b3e88\.system_generated\logs\transcript_full.jsonl'

# Edits made across previous steps (from beginning up to step 1360 before our current turn started)
all_edits = []
with open(p, 'r', encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        idx = d.get('step_index')
        if idx >= 1370: # skip current turn
            continue
        calls = d.get('tool_calls', [])
        for c in calls:
            if c.get('name') == 'edit_file':
                args = c.get('args', {})
                if 'alphabet_classifier.py' in args.get('Path', ''):
                    all_edits.append((idx, args.get('Target'), args.get('Replacement')))

print(f"Found {len(all_edits)} edits to alphabet_classifier.py")

for idx, target, replacement in all_edits:
    if target in text:
        count = text.count(target)
        if count == 1:
            text = text.replace(target, replacement)
            print(f"Applied edit from step {idx} (single match)")
        else:
            text = text.replace(target, replacement)
            print(f"Applied edit from step {idx} (count={count})")
    else:
        print(f"WARNING: Target from step {idx} not found!")

with open('backend/vision/alphabet_classifier.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Finished reconstructing backend/vision/alphabet_classifier.py!")


