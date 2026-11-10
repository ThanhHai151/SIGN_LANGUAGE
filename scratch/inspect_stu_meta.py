import numpy as np
import json

ds = np.load('backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz', allow_pickle=True)
meta = json.loads(str(ds['metadata_json']))
classes = list(ds['classes'])
y = ds['y']

for target in ['S', 'T', 'U']:
    idx = classes.index(target)
    indices = np.where(y == idx)[0]
    print(f"\n--- Letter {target} (class idx {idx}, count={len(indices)}) ---")
    videos = {}
    for i in indices:
        m = meta[i]
        vid = m.get('video_id', 'unknown')
        videos[vid] = videos.get(vid, 0) + 1
    print(f"Video sources: {videos}")

