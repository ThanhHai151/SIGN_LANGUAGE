with open('backend/vision/alphabet_classifier.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, l in enumerate(lines):
    if 'is_s' in l or '"S"' in l:
        if 'Rule' in l or 'rule' in l or 'cand' in l or 'confidence' in l:
            print(f'{i+1}: {l.rstrip()}')
            for j in range(max(0, i-2), min(len(lines), i+15)):
                print(f'  {j+1}: {lines[j].rstrip()}')

