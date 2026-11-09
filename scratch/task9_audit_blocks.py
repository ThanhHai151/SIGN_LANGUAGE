import numpy as np
import json

v2 = np.load('backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz', allow_pickle=True)
clean = np.load('backend/data/alphabet_v2/datasets/vsl_alphabet_static_south_clean_v1.npz', allow_pickle=True)
meta_v2 = json.loads(str(v2['metadata_json']))
meta_clean = json.loads(str(clean['metadata_json']))

ELIGIBLE_INDEX_RANGES = [
    (0, 98),
    (146, 256),
    (277, 324),
    (345, 379),
    (405, 420),
]
keep_indices = []
for start, end in ELIGIBLE_INDEX_RANGES:
    keep_indices.extend(range(start, end + 1))

v2_to_clean = {orig_i: clean_i for clean_i, orig_i in enumerate(keep_indices)}

blocks = []
cur_key = None
start_idx = 0

for i, m in enumerate(meta_v2):
    key = (m.get('label'), m.get('source_letter'), m.get('video_id'), m.get('source_video'), m.get('source_type'))
    if key != cur_key:
        if cur_key is not None:
            blocks.append((cur_key, start_idx, i - 1, i - start_idx))
        cur_key = key
        start_idx = i
blocks.append((cur_key, start_idx, len(meta_v2) - 1, len(meta_v2) - start_idx))

print(f"Total distinct blocks in v2: {len(blocks)}")
header = f"{'No':<3} {'Label':<6} {'SrcLtr':<7} {'VideoID':<10} {'Region':<12} {'Type':<22} {'V2 Range':<10} {'Cnt':<4} {'Clean Status':<22}"
print(header)
print("-" * len(header))

for b_idx, (key, s, e, cnt) in enumerate(blocks):
    lbl, src_ltr, vid, src_vid, stype = key
    if vid.endswith('N'):
        region = 'SOUTH'
    elif vid.endswith('B'):
        region = 'NORTH'
    elif vid.endswith('T'):
        region = 'CENTRAL'
    else:
        region = 'SHARED'

    if s in v2_to_clean:
        clean_s = v2_to_clean[s]
        clean_e = v2_to_clean[e]
        clean_status = f"Clean [{clean_s:>3}, {clean_e:>3}]"
    else:
        clean_status = "EXCLUDED (0 in clean)"
    print(f"{b_idx+1:<3} {str(lbl):<6} {str(src_ltr):<7} {vid:<10} {region:<12} {stype:<22} [{s:>3}, {e:>3}]  {cnt:<4} {clean_status:<22}")

