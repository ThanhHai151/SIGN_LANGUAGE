with open('backend/models/vsl_dictionary.py', encoding='utf-8') as f:
    lines = f.readlines()

for i, l in enumerate(lines):
    if any(k in l for k in ['"P"', '"Q"', '"R"', '"X"', "'P'", "'Q'", "'R'", "'X'"]):
        print("".join(lines[max(0, i-2):min(len(lines), i+8)]))
        print("="*40)

