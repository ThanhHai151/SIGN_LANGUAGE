with open('backend/vision/recognizer.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, l in enumerate(lines):
    if 'def _build_pilot_geometric_payload' in l:
        print(f"=== {i+1}: {l.strip()} ===")
        for j in range(i, min(len(lines), i+60)):
            print(f"  {j+1}: {lines[j].strip()}")
        break

