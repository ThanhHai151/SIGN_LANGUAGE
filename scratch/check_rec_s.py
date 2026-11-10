with open('backend/vision/recognizer.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, l in enumerate(lines):
    if 'classify_geometry' in l or 'classify_frame' in l or '"S"' in l or "'S'" in l:
        print(f"{i+1}: {l.strip()}")
        if 'classify_geometry' in l:
            for j in range(i, min(len(lines), i+40)):
                print(f"  {j+1}: {lines[j].strip()}")
            break

