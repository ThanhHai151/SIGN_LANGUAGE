import sys
import json

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

with open('backend/data/vsl_lexicon.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

ids = ['D0539', 'D0544', 'D0545', 'D0546N']
for item in data['entries']:
    vurl = item.get('video_url') or ''
    for d_id in ids:
        if d_id in vurl:
            print(f"ID: {item.get('id')} | Word: {item.get('word')}")
            print(f"Description: {item.get('description')}")
            print(f"Video URL: {vurl}\n")
