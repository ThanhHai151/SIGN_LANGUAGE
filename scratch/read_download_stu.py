with open('backend/training/download_dataset.py', 'r', encoding='utf-8') as f:
    text = f.read()

lines = text.splitlines()
for i, l in enumerate(lines):
    if any(k in l for k in ['"S"', '"T"', '"U"', "'S'", "'T'", "'U'"]):
        print(f"L{i+1}: {l}")

