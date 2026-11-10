import json
import numpy as np

data = np.load('backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz', allow_pickle=True)
meta = json.loads(str(data['metadata_json']))

blocks = []
current_block = None

for i, m in enumerate(meta):
    lbl = m['label']
    st = m.get('source_type', 'static_video')
    src_let = m.get('source_letter', lbl)
    vid = m['video_id']
    
    # Region
    if vid.endswith('N'):
        region = 'REGION_SOUTH'
    elif vid.endswith('B'):
        region = 'REGION_NORTH'
    elif vid.endswith('T'):
        region = 'REGION_CENTRAL'
    else:
        region = 'REGION_SHARED'
        
    # Semantic
    if st == 'static_video':
        semantic = 'SEMANTIC_CONFIRMED'
    else:
        semantic = 'SEMANTIC_UNCONFIRMED'
        
    # Eligibility
    if region in ('REGION_NORTH', 'REGION_CENTRAL'):
        elig = 'EXCLUDED_REGION'
    elif semantic == 'SEMANTIC_UNCONFIRMED':
        elig = 'PENDING_VERIFICATION'
    elif region == 'REGION_SHARED' and semantic == 'SEMANTIC_CONFIRMED':
        elig = 'ELIGIBLE_SHARED'
    elif region == 'REGION_SOUTH' and semantic == 'SEMANTIC_CONFIRMED':
        elig = 'ELIGIBLE_SOUTH'
    else:
        elig = 'PENDING_VERIFICATION'
        
    key = (lbl, src_let, vid, region, st, semantic, elig)
    
    if current_block is None or current_block['key'] != key:
        if current_block is not None:
            blocks.append(current_block)
        current_block = {
            'key': key,
            'label': lbl,
            'source_letter': src_let,
            'video_id': vid,
            'region': region,
            'source_type': st,
            'semantic': semantic,
            'eligibility': elig,
            'start_idx': i,
            'end_idx': i,
            'count': 1
        }
    else:
        current_block['end_idx'] = i
        current_block['count'] += 1

if current_block is not None:
    blocks.append(current_block)

print(f"Total contiguous blocks found: {len(blocks)}")
print("="*125)
print(f"{'#':2s} | {'Label':5s} | {'SrcLet':6s} | {'Start':5s} | {'End':5s} | {'Count':5s} | {'Video ID':8s} | {'Region':14s} | {'Source Type':20s} | {'Eligibility':20s}")
print("-"*125)
total_count = 0
for idx, b in enumerate(blocks):
    print(f"{idx+1:2d} | {b['label']:5s} | {b['source_letter']:6s} | {b['start_idx']:5d} | {b['end_idx']:5d} | {b['count']:5d} | {b['video_id']:8s} | {b['region']:14s} | {b['source_type']:20s} | {b['eligibility']:20s}")
    total_count += b['count']
print("="*125)
print(f"TOTAL SAMPLES: {total_count}")

