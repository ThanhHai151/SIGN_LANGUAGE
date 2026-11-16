with open('backend/vision/recognizer.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, l in enumerate(lines):
    if 'history' in l or 'clear()' in l or 'smoothing' in l or 'smooth' in l:
        if 'letter' in l or 'recent' in l or 'buffer' in l or 'R' in l or 'X' in l or 'S' in l:
            print(f"{i+1}: {l.strip()}")
            for j in range(max(0, i-2), min(len(lines), i+8)):
                print(f"  {j+1}: {lines[j].strip()}")

