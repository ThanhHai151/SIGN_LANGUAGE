with open('backend/vision/recognizer.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, l in enumerate(lines):
    if 1800 <= i <= 2150:
        if 'return {' in l or '"sign":' in l or '"confirmed":' in l:
            print(f"{i+1}: {l.strip()}")

