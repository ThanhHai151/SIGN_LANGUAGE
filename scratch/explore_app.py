with open('backend/app.py', 'r', encoding='utf-8') as f:
    text = f.read()

lines = text.splitlines()
print("app.py lines:", len(lines))
for i, line in enumerate(lines):
    if any(k in line.lower() for k in ['recognizer', 'websocket', 'mode', 'letters', 'camera']):
        print(f"L{i+1}: {line.strip()[:100]}")

