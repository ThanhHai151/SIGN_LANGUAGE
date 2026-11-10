with open('backend/vision/recognizer.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print("--- Lines 1700 to 1735 ---")
for i in range(1699, 1735):
    print(f"{i+1}: {lines[i].rstrip()}")

