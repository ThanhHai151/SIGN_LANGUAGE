with open('backend/vision/recognizer.py', 'r', encoding='utf-8') as f:
    text = f.read()

lines = text.splitlines()
for i, l in enumerate(lines):
    if any(k in l.lower() for k in ['smooth', 'deque', 'buffer', 'temporal', 'vote', 'window']):
        if i < 250 or i > 1200:
            print(f"L{i+1}: {l.strip()[:100]}")

