with open('src/components/CameraSignRecognizer.jsx', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if 'test(' in line or "requestMode === 'letters'" in line or "conversionMode === 'letters'" in line:
            print(f"{i+1}: {line.strip()}")

