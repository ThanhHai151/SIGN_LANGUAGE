with open('backend/tests/test_alphabet_classifier.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, l in enumerate(lines):
    if 'def test_' in l and ('_s' in l or '_t_u_s' in l or 'qrsx' in l or 'distinct' in l):
        print(f"{i+1}: {l.strip()}")

