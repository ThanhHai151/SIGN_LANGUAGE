with open('backend/vision/alphabet_classifier.py', 'r', encoding='utf-8') as f:
    text = f.read()

lines = text.splitlines()
print(f"Total lines: {len(lines)}")
for i, l in enumerate(lines):
    if 'def evaluate_rule_t_diagnostics' in l:
        print(f"L{i+1}: {l}")

