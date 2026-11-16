with open('backend/vision/alphabet_classifier.py', 'r', encoding='utf-8') as f:
    text = f.read()

for line in text.splitlines():
    if line.startswith('S_') or line.startswith('T_') or line.startswith('U_'):
        print(line)

