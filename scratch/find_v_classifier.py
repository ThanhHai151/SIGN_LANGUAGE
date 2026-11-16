with open('backend/vision/alphabet_classifier.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, l in enumerate(lines, 1):
    low = l.lower()
    if 'letter_v' in low or '\"v\"' in low or "'v'" in low or 'def ' in low:
        if 'def ' in low:
            print(f"{i}: {l.strip()}")
        elif 'letter_v' in low or '\"v\"' in low or "'v'" in low:
            print(f"  {i}: {l.strip()}")

