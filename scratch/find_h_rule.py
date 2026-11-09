with open('backend/vision/alphabet_classifier.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, l in enumerate(lines, 1):
    low = l.lower()
    if '\"h\"' in l or "'h'" in l or 'chữ h' in low or 'rule_h' in low or 'h_diag' in low:
        print(f"{i}: {l.strip()}")

