import json

with open('backend/data/vsl_lexicon.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

entries = data['entries']
orig_count = len(entries)

purged_entries = []
purged_gt_ids = []
removed_gt_variants = 0

for e in entries:
    eid = e.get('id', '')
    if eid.startswith('gt-'):
        purged_gt_ids.append(eid)
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
    
    # If e.get('region') == 'GT', clean it
    if e.get('region') == 'GT':
        e['region'] = ''
        
    purged_entries.append(e)

print(f'Original entries: {orig_count}')
print(f'Purged GT entries: {len(purged_gt_ids)}')
print(f'Remaining entries: {len(purged_entries)}')
print(f'Removed GT variants from remaining entries: {removed_gt_variants}')

# Check if any remaining entry has gt- ID or supabase URL or GT region
any_gt_id = [e['id'] for e in purged_entries if e['id'].startswith('gt-')]
any_supa = [e['id'] for e in purged_entries if 'supabase.co' in (e.get('video_url') or '') or any('supabase.co' in (v.get('url') or '') for v in e.get('variants', []))]
any_gt_reg = [e['id'] for e in purged_entries if e.get('region') == 'GT' or any(v.get('region') == 'GT' for v in e.get('variants', []))]

print(f'Any remaining gt- id: {len(any_gt_id)}')
print(f'Any remaining supabase URL: {len(any_supa)}')
print(f'Any remaining GT region: {len(any_gt_reg)}')
