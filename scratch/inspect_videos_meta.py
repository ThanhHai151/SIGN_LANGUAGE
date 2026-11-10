import json

with open('backend/data/alphabet_v2/metadata/videos_metadata.json', 'r', encoding='utf-8') as f:
    vids = json.load(f)

for v in vids:
    if any(k in v.get('video_id', '') for k in ['D0512', 'D0513', 'D0514', 'D0515']):
        print(v)

