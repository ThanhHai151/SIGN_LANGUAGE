with open("src/utils/handPoses.js", "r", encoding="utf-8") as f:
    lines = f.readlines()

in_pose = False
for i, line in enumerate(lines):
    for let in ["P", "Q", "R", "X"]:
        if f"name: 'Chữ {let}" in line or f"name: '{let}'" in line or f"id: '{let}'" in line:
            print(f"--- Line {i+1}: {line.strip()} ---")
            for j in range(max(0, i-5), min(len(lines), i+25)):
                print(lines[j], end="")
            print("\n" + "="*40)
