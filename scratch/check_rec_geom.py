with open('backend/vision/recognizer.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i in range(570, 670):
    print(f"{i+1}: {lines[i].rstrip()}")

