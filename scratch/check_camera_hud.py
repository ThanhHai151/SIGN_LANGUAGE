with open('src/components/CameraSignRecognizer.jsx', 'r', encoding='utf-8') as f:
    text = f.read()

lines = text.splitlines()
for i, l in enumerate(lines):
    if any(k in l.lower() for k in ['result.sign', 'display_sign', 'confirmed_sign', 'handlesignresult', 'current_sign', 'final_sign']):
        print(f"L{i+1}: {l.strip()[:100]}")

