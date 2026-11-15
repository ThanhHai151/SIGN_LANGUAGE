import json
import re

with open('backend/data/vsl_lexicon.json', encoding='utf-8') as f:
    d = json.load(f)

targets = ['a', 'ă', 'â', 'e', 'ê', 'o', 'ô', 'ơ', 'u', 'ư', 'd', 'đ']
entries = d.get('entries', [])
print(f"Total entries in lexicon: {len(entries)}")

found = []
for e in entries:
    w = e.get('word', '').lower().strip()
    keys = [k.lower().strip() for k in e.get('keys', [])]
    v_url = e.get('video_url') or ''
    if w in targets or any(k in targets for k in keys) or any(v in v_url for v in ['D0507', 'D0508', 'D0490', 'D0491', 'D0497', 'D0495', 'D0515']):
        found.append(e)

print(f"Found {len(found)} matching entries:")
for e in found:
    print("-" * 60)
    print(f"ID: {e.get('id')} | Word: {e.get('word')} | Keys: {e.get('keys')}")
    print(f"How_to: {e.get('how_to')}")
    print(f"Video URL: {e.get('video_url')}")
    print(f"Variants: {e.get('variants')}")
