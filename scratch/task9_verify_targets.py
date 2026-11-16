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

targets = ['D0498B', 'D0500B', 'D0509B', 'D0497B', 'D0507B']

print("=== XÁC MINH 5 MÃ VIDEO MỤC TIÊU TRONG NPZ GỐC ===")
for t in targets:
    matching = [i for i, m in enumerate(meta_v2) if m.get('video_id') == t]
    first = meta_v2[matching[0]]
    last = meta_v2[matching[-1]]
    in_clean = [v2_to_clean[i] for i in matching if i in v2_to_clean]
    vid = first.get('video_id')
    reg = "SOUTH" if vid.endswith("N") else ("NORTH" if vid.endswith("B") else ("CENTRAL" if vid.endswith("T") else "SHARED"))
    print(f"Mã video: {t}")
    print(f"  - Số mẫu: {len(matching)} mẫu | Dải index trong v2: [{matching[0]}, {matching[-1]}]")
    print(f"  - Label: '{first.get('label')}' | source_letter: '{first.get('source_letter')}'")
    print(f"  - Vùng miền (tính từ video_id): {reg}")
    print(f"  - source_type: '{first.get('source_type')}'")
    print(f"  - source_video: '{first.get('source_video')}'")
    print(f"  - Trạng thái trong dataset sạch: {'CÓ ' + str(len(in_clean)) if in_clean else 'LOẠI BỎ HOÀN TOÀN (0 mẫu)'}")
    print()

