import numpy as np
import json

ds2 = np.load('backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz', allow_pickle=True)
meta2 = json.loads(str(ds2['metadata_json']))
classes2 = list(ds2['classes'])
y2 = ds2['y']
lms2 = ds2['landmarks']

with open('backend/data/alphabet_v2/static_landmarks/T.json', 'r', encoding='utf-8') as f:
    js = json.load(f)

print("T.json samples:")
for s in js['samples']:
    if s.get('landmarks'):
        print(f"  frame={s['frame_index']}, phase={s['phase']}, lm0={s['landmarks'][0]}")

print("\nstatic_v2 T samples:")
t_idx = classes2.index('T')
for i in np.where(y2 == t_idx)[0]:
    m = meta2[i]
    print(f"  frame={m['frame_index']}, quality={m['quality_status']}, lm0={lms2[i][0]}")

