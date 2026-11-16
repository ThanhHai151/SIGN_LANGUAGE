with open('backend/vision/recognizer.py', 'r', encoding='utf-8') as f:
    text = f.read()

lines = text.splitlines()
for i in range(540, 680):
    print(f"L{i+1}: {lines[i]}")

