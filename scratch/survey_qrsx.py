import numpy as np
import json
import os

# 1. Inspect npz
npz_path = 'backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz'
data = np.load(npz_path, allow_pickle=True)
classes = [str(c) for c in data['classes']]
y = data['y']
print("Classes list:", classes)

for let in ['Q', 'R', 'S', 'X']:
    idx = classes.index(let)
    cnt = int(np.sum(y == idx))
    print(f"NPZ: Letter {let} -> index {idx}, count = {cnt}")

# 2. Inspect static_landmarks JSON
for let in ['Q', 'R', 'S', 'X']:
    json_path = f'backend/data/alphabet_v2/static_landmarks/{let}.json'
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            jdata = json.load(f)
        samps = jdata.get('samples', [])
        valid_samps = [s for s in samps if s.get('landmarks')]
        stable_samps = [s for s in valid_samps if s.get('phase') == 'stable_pose']
        print(f"JSON: Letter {let} -> video_id = {jdata.get('video_id')}, total_samples = {len(samps)}, valid = {len(valid_samps)}, stable = {len(stable_samps)}")
    else:
        print(f"JSON: Letter {let} -> NOT FOUND at {json_path}")

# 3. Check video files in backend/data/media_cache/videos
video_dir = 'backend/data/media_cache/videos'
print("\nChecking video files in", video_dir)
for f in os.listdir(video_dir):
    if f.endswith('.mp4'):
        print("Video:", f)
