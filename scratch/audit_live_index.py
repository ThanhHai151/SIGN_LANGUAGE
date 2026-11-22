import json
from pathlib import Path
import re
import numpy as np

with open('backend/data/vsl_lexicon.json', 'r', encoding='utf-8') as f:
    lex = json.load(f)

by_id = {e['id']: e for e in lex['entries']}

data = np.load('backend/data/qipedc_reference/live_index.npz')
uni_ids = [str(x) for x in data['uni_ids']]
bi_ids = [str(x) for x in data['bi_ids']]
all_ids = uni_ids + bi_ids

print(f"Total IDs in live_index.npz: {len(all_ids)} (uni: {len(uni_ids)}, bi: {len(bi_ids)})")

not_in_lex = [eid for eid in all_ids if eid not in by_id]
print(f"IDs not in lexicon: {len(not_in_lex)}")

regions = {}
vid_suffixes = {'national': 0, 'N': 0, 'B': 0, 'T': 0, 'GT': 0, 'none': 0, 'other': 0}

vid_re = re.compile(r'/([^/]+)\.mp4')

for eid in all_ids:
    e = by_id.get(eid)
    if not e:
        continue
    reg = e.get('region')
    regions[reg] = regions.get(reg, 0) + 1
    
    vurl = e.get('video_url') or ''
    m = vid_re.search(vurl)
    vid = m.group(1) if m else ''
    
    if re.search(r'B$', vid, re.IGNORECASE) and not vid.startswith('sign_'):
        vid_suffixes['B'] += 1
    elif re.search(r'T$', vid, re.IGNORECASE) and not vid.startswith('sign_'):
        vid_suffixes['T'] += 1
    elif re.match(r'^[A-Z]\d+N$', vid):
        vid_suffixes['N'] += 1
    elif re.match(r'^[A-Z]\d+$', vid):
        vid_suffixes['national'] += 1
    elif 'sign_' in vid:
        vid_suffixes['GT'] += 1
    elif not vid:
        vid_suffixes['none'] += 1
    else:
        vid_suffixes['other'] += 1

print("=== REGION DISTRIBUTION IN LIVE_INDEX.NPZ ===")
for r, count in regions.items():
    print(f"  region={r}: {count}")

print("\n=== VIDEO_URL SUFFIX DISTRIBUTION IN LIVE_INDEX.NPZ ===")
for s, count in vid_suffixes.items():
    print(f"  {s}: {count}")

# Check if there are any duplicate IDs between uni and bi
overlap = set(uni_ids) & set(bi_ids)
print(f"\nOverlap between uni and bi: {len(overlap)}")

# Check what the 173 B-suffix files in media_cache are!
mc = Path('backend/data/media_cache')
b_vids = [f.stem for f in mc.glob('*B.mp4')]
print(f"\nTotal B videos in media_cache: {len(b_vids)}")

# Does ANY entry in lexicon mention these B videos?
entries_with_b_vid = []
for e in lex['entries']:
    # check video_url
    urls = [e.get('video_url') or ''] + [v.get('url') or '' for v in e.get('variants', [])]
    for u in urls:
        for b in b_vids:
            if b in u:
                entries_with_b_vid.append((e['id'], b, u))

print(f"Entries in lexicon mentioning B videos: {len(entries_with_b_vid)}")
