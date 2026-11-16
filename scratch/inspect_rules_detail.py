with open('backend/vision/alphabet_classifier.py', 'r', encoding='utf-8') as f:
    text = f.read()

for l in ['G', 'H', 'P', 'T', 'X']:
    idx = text.find(f'matches.append(("{l}"')
    if idx == -1:
        idx = text.find(f"matches.append(('{l}'")
    print(f"Letter {l}: rule found at char {idx}")
    if idx != -1:
        # find comment before it
        sec_start = text.rfind('# ---', 0, idx)
        print(text[sec_start:idx+150])
        print("="*60)

