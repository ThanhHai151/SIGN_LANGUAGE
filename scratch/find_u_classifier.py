with open('backend/vision/alphabet_classifier.py', 'r', encoding='utf-8') as f:
    text = f.read()

lines = text.splitlines()
for i, line in enumerate(lines):
    if 'Chữ U' in line or '"U"' in line or "'U'" in line:
        print(f"L{i+1}: {line}")

