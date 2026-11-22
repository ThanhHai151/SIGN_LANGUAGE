import json
import shutil
import os

src_path = 'backend/data/vsl_lexicon.json'
bak_path = 'backend/data/vsl_lexicon.json.bak'

if not os.path.exists(bak_path):
    shutil.copyfile(src_path, bak_path)
    print('Created backup at', bak_path)

with open(src_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

entries = data['entries']
orig_count = len(entries)

purged_entries = []
purged_gt = []
removed_gt_variants = 0

for e in entries:
    eid = e.get('id', '')
    if eid.startswith('gt-'):
        purged_gt.append(e)
        continue
    
    # Clean variants
    cleaned_variants = []
    for v in e.get('variants', []):
        if isinstance(v, dict):
            if v.get('region') == 'GT' or 'supabase.co' in (v.get('url') or ''):
                removed_gt_variants += 1
                continue
        cleaned_variants.append(v)
    e['variants'] = cleaned_variants
    
    if e.get('region') == 'GT':
        e['region'] = ''
        
    purged_entries.append(e)

print(f'Original entries: {orig_count}')
print(f'Purged GT entries: {len(purged_gt)}')
print(f'Remaining entries: {len(purged_entries)}')
print(f'Removed GT variants from remaining entries: {removed_gt_variants}')

# Update meta
meta = data.get('meta', {})
meta['total'] = len(purged_entries)
meta['sstcn_linked'] = sum(1 for e in purged_entries if e.get('sstcn_index') is not None)
video_count = sum(1 for e in purged_entries if e.get('video_url')) + sum(len(e.get('variants', [])) for e in purged_entries)
meta['total_videos'] = video_count
if 'sources' in meta and 'Giáo trình VSL (102 từ)' in meta['sources']:
    del meta['sources']['Giáo trình VSL (102 từ)']

data['entries'] = purged_entries
data['meta'] = meta

with open(src_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print('Successfully written updated vsl_lexicon.json!')
