import numpy as np
import sys
sys.path.insert(0, '.')
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

clf = VSLAlphabetClassifier()

data = np.load('backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz', allow_pickle=True)
y = data['y']
classes = list(data['classes'])
landmarks = data['landmarks']

for target_name in ['Q', 'R', 'S', 'X']:
    idx = classes.index(target_name)
    target_lms = landmarks[y == idx]
    print("=" * 80)
    print(f"ANALYZE DATASET SAMPLES FOR LETTER {target_name} (Class {idx}, {len(target_lms)} samples)")
    print("=" * 80)
    
    feats = []
    for i, lm in enumerate(target_lms):
        feat = clf.extract_features(lm.tolist())
        if not feat:
            print(f"Sample {i}: extract_features returned None")
            continue
        feats.append(feat)
        
    print(f"Successfully extracted {len(feats)}/{len(target_lms)} feature sets.")
    
    # Analyze key metrics
    keys = [
        'd_IW', 'd_MW', 'd_RW', 'd_PW', 'd_TW',
        'd_TI', 'd_TM', 'd_IM', 'diff_x_IM', 'ang_IM_dir',
        'idx_straight', 'mid_straight', 'ring_straight', 'pinky_straight',
        'index_curl', 'middle_curl', 'ring_curl', 'pinky_curl',
        'idx_tip_y', 'mid_tip_y', 'thumb_lat', 'palm_scale'
    ]
    
    for k in keys:
        vals = [f[k] for f in feats if k in f]
        if vals:
            print(f"  {k:16s}: min={np.min(vals):.3f}, max={np.max(vals):.3f}, mean={np.mean(vals):.3f}, std={np.std(vals):.3f}")

