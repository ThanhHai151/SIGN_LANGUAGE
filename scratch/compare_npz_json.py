import numpy as np
import json

ds2 = np.load('backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz', allow_pickle=True)
classes2 = list(ds2['classes'])
y2 = ds2['y']
lms2 = ds2['landmarks']
meta2 = json.loads(str(ds2['metadata_json']))

print("Classes in static_v2:", classes2)
for letter in ['T', 'U', 'S']:
    idx = classes2.index(letter)
    indices = np.where(y2 == idx)[0]
    print(f"\nLetter {letter}: {len(indices)} samples in static_v2")
    # compare with static_landmarks/{letter}.json
    with open(f'backend/data/alphabet_v2/static_landmarks/{letter}.json', 'r', encoding='utf-8') as f:
        js = json.load(f)
        valid_js = [s for s in js['samples'] if s.get('landmarks') is not None]
        print(f"  Valid samples in {letter}.json: {len(valid_js)}")
        # Check first sample
        if len(valid_js) > 0 and len(indices) > 0:
            lm_npz = lms2[indices[0]]
            lm_js = np.array(valid_js[0]['landmarks'])
            diff = np.max(np.abs(lm_npz - lm_js))
            print(f"  Max abs diff sample 0: {diff:.6f}")

