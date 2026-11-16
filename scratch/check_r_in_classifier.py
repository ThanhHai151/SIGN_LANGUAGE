with open('backend/vision/alphabet_classifier.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()
for i, line in enumerate(lines):
    if 'Rule R' in line or 'evaluate_rule_r' in line or 'R_CROSS' in line or 'Chữ R' in line or 'matched_letter == "R"' in line:
        print(f'{i+1}: {line.strip()}')

