import re

with open('backend/vision/alphabet_classifier.py', 'r', encoding='utf-8') as f:
    text = f.read()

print("Length of alphabet_classifier.py:", len(text.splitlines()))

# Find all occurrences of letter checks or evaluation logic
patterns = [
    r'def\s+[a-zA-Z0-9_]+\(',
    r'# --- [A-Z]',
    r'match_letter',
    r'scores\[',
    r'candidates\.'
]
for p in patterns:
    matches = list(re.finditer(p, text))
    print(f"Pattern {p}: {len(matches)} matches")

print("\n--- Searching for T, U, S in alphabet_classifier.py ---")
for i, line in enumerate(text.splitlines()):
    if any(q in line for q in ['"T"', "'T'", '"U"', "'U'", '"S"', "'S'", 'letter_t', 'letter_u', 'letter_s', 'rule_t', 'rule_u', 'rule_s']):
        print(f"L{i+1}: {line.strip()[:100]}")

