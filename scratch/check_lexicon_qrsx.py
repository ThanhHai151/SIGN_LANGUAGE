import json

with open('backend/data/vsl_lexicon.json', 'r', encoding='utf-8') as f:
    lex = json.load(f)

entries = lex.get('entries', [])
print("Total entries:", len(entries))

for target_id in ['D0510', 'D0511', 'D0512', 'D0517B']:
    found = False
    for e in entries:
        v_url = e.get('video_url') or ''
        how_to = e.get('how_to') or ''
        word = e.get('word', '')
        if target_id in v_url or target_id in str(e):
            print(f"\n--- Found {target_id} in Lexicon ---")
            print(f"Word: {word}")
            print(f"Definition: {e.get('definition')}")
            print(f"How_to: {how_to}")
            print(f"Video URL: {v_url}")
            found = True
            break
    if not found:
        print(f"Not found {target_id} in lexicon entries")
