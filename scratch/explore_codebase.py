import re

with open('backend/vision/alphabet_classifier.py', 'r', encoding='utf-8') as f:
    text = f.read()

print("=== alphabet_classifier.py matches ===")
for m in re.finditer(r'def (?:_eval|_check|is_|classify)[a-zA-Z0-9_]*', text):
    print(m.group(0))

print("\n=== recognizer.py matches ===")
with open('backend/vision/recognizer.py', 'r', encoding='utf-8') as f:
    text_rec = f.read()

for m in re.finditer(r'def (?:_eval|_check|is_|classify|_recognize)[a-zA-Z0-9_]*', text_rec):
    print(m.group(0))

