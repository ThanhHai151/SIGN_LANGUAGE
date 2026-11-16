with open('backend/vision/alphabet_classifier.py', 'r', encoding='utf-8') as f:
    text = f.read()

for l in ['G', 'H', 'P', 'X']:
    idx = 0
    found = []
    while True:
        idx = text.find('matches.append', idx)
        if idx == -1:
            break
        snippet = text[idx:idx+80]
        if f'"{l}"' in snippet or f"'{l}'" in snippet:
            found.append((idx, snippet))
        idx += 1
    print(f"Letter {l}: {len(found)} occurrences in matches.append")
    for pos, snip in found:
        print(f"   pos {pos}: {snip.strip()}")

