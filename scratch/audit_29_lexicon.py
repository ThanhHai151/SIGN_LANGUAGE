import json, os

LETTERS_29 = [
    'A', 'Ă', 'Â', 'B', 'C', 'D', 'Đ', 'E', 'Ê', 'G', 'H', 'I', 'K', 'L', 'M',
    'N', 'O', 'Ô', 'Ơ', 'P', 'Q', 'R', 'S', 'T', 'U', 'Ư', 'V', 'X', 'Y'
]

with open('backend/data/vsl_lexicon.json', 'r', encoding='utf-8') as f:
    entries = json.load(f)['entries']

print("=== 29 LETTERS IN VSL_LEXICON.JSON ===")
for e in entries:
    w = e.get('word', '').upper()
    if w in LETTERS_29:
        eid = e.get('id')
        vid = e.get('video')
        reg = e.get('region')
        regs = e.get('regions')
        vars = e.get('variants')
        print(f"{w} (id={eid}): video={vid}, region={reg}, regions={regs}, variants={vars}")

