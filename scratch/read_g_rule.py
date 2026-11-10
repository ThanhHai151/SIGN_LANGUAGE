with open('backend/vision/alphabet_classifier.py', encoding='utf-8') as f:
    lines = f.readlines()

for i, l in enumerate(lines):
    if '"G"' in l and 'matches.append' in l:
        print("".join(lines[max(0, i-30):min(len(lines), i+10)]))

