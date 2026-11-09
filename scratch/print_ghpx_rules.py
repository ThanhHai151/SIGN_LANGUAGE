with open('backend/vision/alphabet_classifier.py', 'r', encoding='utf-8') as f:
    text = f.read()

for l in ['G', 'H', 'P', 'X']:
    idx = text.find(f'matches.append((\n                    "{l}"')
    if idx == -1:
        idx = text.find(f'matches.append((\n                "{l}"')
    print(f"=== Letter {l} (pos {idx}) ===")
    start = text.rfind('# -------------------------------------------------------------------', 0, idx)
    print(text[start:idx+150])

