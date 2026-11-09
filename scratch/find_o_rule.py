with open("backend/vision/alphabet_classifier.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

for i, l in enumerate(lines):
    if "Rule 6" in l or "Candidate-03" in l or "CHỮ O" in l or " Quy tắc 6" in l or "quy tắc 6" in l.lower() or "chữ o" in l.lower():
        print(f"{i+1}: {l.strip()}")

