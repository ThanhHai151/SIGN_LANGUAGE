with open('backend/vision/alphabet_classifier.py', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if any(f'"{ch}"' in line or f"'{ch}'" in line for ch in ['M', 'N']):
            print(f"{i+1}: {line.strip()}")

