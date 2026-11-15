import json

LETTERS_29 = [
    'A', 'Ă', 'Â', 'B', 'C', 'D', 'Đ', 'E', 'Ê', 'G', 'H', 'I', 'K', 'L', 'M',
    'N', 'O', 'Ô', 'Ơ', 'P', 'Q', 'R', 'S', 'T', 'U', 'Ư', 'V', 'X', 'Y'
]

with open('backend/data/vsl_lexicon.json', 'r', encoding='utf-8') as f:
    entries = json.load(f)['entries']

entries_by_letter = {}
for e in entries:
    w = e.get('word', '').upper()
    if w in LETTERS_29:
        entries_by_letter[w] = e

print(f"{'Letter':<4} | {'Current Meta Video':<18} | {'Lexicon Available Variants (region: filename)'}")
print("-" * 80)

with open('backend/data/alphabet_v2/metadata/videos_metadata.json', 'r', encoding='utf-8') as f:
    v_meta = json.load(f)

for l in LETTERS_29:
    meta = v_meta.get(l, {})
    curr_vid = meta.get('filename') or meta.get('video_id')
    e = entries_by_letter.get(l, {})
    variants = e.get('variants', [])
    var_list = []
    for v in variants:
        reg = v.get('region', '')
        url = v.get('url', '')
        fname = url.split('/')[-1] if url else ''
        if 'qipedc' in url:
            var_list.append(f"{reg or 'National'}: {fname}")
    print(f"{l:<4} | {str(curr_vid):<18} | {', '.join(var_list)}")

