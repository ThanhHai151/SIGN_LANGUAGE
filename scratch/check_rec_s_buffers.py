with open('backend/vision/recognizer.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print("--- Lines 1495 to 1525 ---")
for i in range(1494, 1525):
    print(f"{i+1}: {lines[i].rstrip()}")

print("\n--- Lines 1730 to 1760 ---")
for i in range(1729, 1760):
    print(f"{i+1}: {lines[i].rstrip()}")

