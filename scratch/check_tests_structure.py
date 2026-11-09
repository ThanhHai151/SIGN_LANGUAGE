with open('backend/tests/test_alphabet_classifier.py', 'r', encoding='utf-8') as f:
    text = f.read()

lines = text.splitlines()
print(f"Total lines in test_alphabet_classifier.py: {len(lines)}")
for i, l in enumerate(lines):
    if l.startswith('def test_') or l.startswith('class Test'):
        print(f"L{i+1}: {l}")

