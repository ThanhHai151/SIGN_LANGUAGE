import os

found = []
for root, dirs, files in os.walk('.'):
    if '.git' in root or 'node_modules' in root:
        continue
    for f in files:
        if f.endswith('.jsonl') or (f.endswith('.json') and 'pilot' in f):
            p = os.path.join(root, f)
            try:
                with open(p, 'r', encoding='utf-8', errors='ignore') as fp:
                    content = fp.read()
                    if '"expected_sign": "O"' in content or '"expected_sign":"O"' in content:
                        found.append(p)
            except Exception as e:
                pass

print("Files with expected_sign == 'O':")
for p in found:
    print(" -", p)

