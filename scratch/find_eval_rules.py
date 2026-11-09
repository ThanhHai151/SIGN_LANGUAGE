with open('backend/vision/alphabet_classifier.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

in_eval = False
for i, l in enumerate(lines):
    if 'def evaluate_rules' in l:
        in_eval = True
        print(f"=== {i+1}: {l.strip()} ===")
    if in_eval:
        if 'Rule' in l or 'cand' in l or 'is_' in l or 'return' in l or 'elif' in l:
            print(f"{i+1}: {l.strip()}")
        if 'def ' in l and not 'def evaluate_rules' in l:
            in_eval = False

