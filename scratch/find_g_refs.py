with open('backend/vision/alphabet_classifier.py', encoding='utf-8') as f:
    lines = f.readlines()

for i, l in enumerate(lines):
    if '"G"' in l or "'G'" in l:
        print(f"Line {i}: {l.strip()}")

