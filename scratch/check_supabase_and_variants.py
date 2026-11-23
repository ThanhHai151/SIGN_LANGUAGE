import json

with open('backend/data/vsl_lexicon.json', 'r', encoding='utf-8') as f:
    lex = json.load(f)

entries = lex['entries']
has_gt_variant = []
has_supabase_url = []
for e in entries:
    is_gt = e['id'].startswith('gt-')
    # check video_url
    if 'supabase.co' in (e.get('video_url') or ''):
        has_supabase_url.append((e['id'], is_gt, 'video_url', e['video_url']))
    # check variants
    for v in e.get('variants', []):
        if isinstance(v, dict):
            if v.get('region') == 'GT':
                has_gt_variant.append((e['id'], is_gt, v))
            if 'supabase.co' in (v.get('url') or ''):
                has_supabase_url.append((e['id'], is_gt, 'variant_url', v.get('url')))

print(f'Entries with region==GT variant: {len(has_gt_variant)}')
print(f'  GT entries with GT variant: {len([x for x in has_gt_variant if x[1]])}')
print(f'  Non-GT entries with GT variant: {len([x for x in has_gt_variant if not x[1]])}')
non_gt_variants = [x for x in has_gt_variant if not x[1]]
for x in non_gt_variants[:10]:
    print('   ', x)

print(f'Entries with supabase URL: {len(has_supabase_url)}')
print(f'  GT entries with supabase URL: {len([x for x in has_supabase_url if x[1]])}')
print(f'  Non-GT entries with supabase URL: {len([x for x in has_supabase_url if not x[1]])}')
non_gt_supa = [x for x in has_supabase_url if not x[1]]
for x in non_gt_supa[:10]:
    print('   ', x)
