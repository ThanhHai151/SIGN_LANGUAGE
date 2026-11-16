import json
import numpy as np

# Load NPZ
data = np.load('backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz', allow_pickle=True)
meta = json.loads(str(data['metadata_json']))

# Verify every sample
sample_details = []
for i, m in enumerate(meta):
    lbl = m['label']
    st = m.get('source_type', 'static_video')
    src_let = m.get('source_letter', lbl)
    vid = m['video_id']
    fidx = m['frame_index']
    is_rep = m.get('is_representative', True)
    
    if vid.endswith('N'):
        region = 'REGION_SOUTH'
    elif vid.endswith('B'):
        region = 'REGION_NORTH'
    elif vid.endswith('T'):
        region = 'REGION_CENTRAL'
    else:
        region = 'REGION_SHARED'
        
    if st == 'static_video':
        semantic = 'SEMANTIC_CONFIRMED'
    else:
        semantic = 'SEMANTIC_UNCONFIRMED'
        
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
        
    sample_details.append({
        'idx': i,
        'label': lbl,
        'source_letter': src_let,
        'source_type': st,
        'video_id': vid,
        'frame_index': fidx,
        'is_representative': is_rep,
        'region': region,
        'semantic': semantic,
        'eligibility': elig
    })

# Check totals
counts = {
    'ELIGIBLE_SHARED': sum(1 for s in sample_details if s['eligibility'] == 'ELIGIBLE_SHARED'),
    'ELIGIBLE_SOUTH': sum(1 for s in sample_details if s['eligibility'] == 'ELIGIBLE_SOUTH'),
    'PENDING_VERIFICATION': sum(1 for s in sample_details if s['eligibility'] == 'PENDING_VERIFICATION'),
    'EXCLUDED_REGION': sum(1 for s in sample_details if s['eligibility'] == 'EXCLUDED_REGION'),
    'EXCLUDED_SEMANTIC': sum(1 for s in sample_details if s['eligibility'] == 'EXCLUDED_SEMANTIC'),
}

print("Counts:", counts)
print("Sum:", sum(counts.values()))

# Verify static_video breakdown
static_samples = [s for s in sample_details if s['source_type'] == 'static_video']
static_shared = sum(1 for s in static_samples if s['region'] == 'REGION_SHARED')
static_south = sum(1 for s in static_samples if s['region'] == 'REGION_SOUTH')
static_north = sum(1 for s in static_samples if s['region'] == 'REGION_NORTH')
static_central = sum(1 for s in static_samples if s['region'] == 'REGION_CENTRAL')

print(f"Static video samples: {len(static_samples)}")
print(f"  Shared: {static_shared}")
print(f"  South: {static_south}")
print(f"  North: {static_north}")
print(f"  Central: {static_central}")

