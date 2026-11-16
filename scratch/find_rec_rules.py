with open('backend/vision/recognizer.py', 'r', encoding='utf-8') as f:
    text = f.read()

lines = text.splitlines()
print("Total lines in recognizer.py:", len(lines))
for i, line in enumerate(lines):
    if any(k in line.lower() for k in ['alphabet', 'classifier', 'geometric', 'rule_layer', 'vslalphabet']):
        print(f"L{i+1}: {line.strip()[:100]}")

