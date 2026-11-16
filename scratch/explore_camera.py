with open('src/components/CameraSignRecognizer.jsx', 'r', encoding='utf-8') as f:
    text = f.read()

lines = text.splitlines()
print("CameraSignRecognizer lines:", len(lines))
for i, line in enumerate(lines):
    if any(k in line.lower() for k in ['letters', 'mode', 'recognize_landmarks', 'smooth', 'fps']):
        if i < 200:
            print(f"L{i+1}: {line.strip()[:100]}")

