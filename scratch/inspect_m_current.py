import re

with open('backend/vision/alphabet_classifier.py', 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f):
        if re.search(r'\b(M|N)\b|chữ M|chữ N', line):
            print(f'{idx+1}: {line.strip()}')

