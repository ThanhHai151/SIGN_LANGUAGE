import sys
sys.path.insert(0, ".")
import json
from backend.vision.alphabet_classifier import alphabet_classifier

# I:
data_i = json.load(open('backend/data/alphabet_v2/static_landmarks/I.json', encoding='utf-8'))
samples_i = [s for s in data_i['samples'] if 14 <= s['frame_index'] <= 40 and s['landmarks'] is not None]
feats_i = [alphabet_classifier.extract_features(s['landmarks']) for s in samples_i]

# Y:
data_y = json.load(open('backend/data/alphabet_v2/static_landmarks/Y.json', encoding='utf-8'))
samples_y = [s for s in data_y['samples'] if 16 <= s['frame_index'] <= 38 and s['landmarks'] is not None]
feats_y = [alphabet_classifier.extract_features(s['landmarks']) for s in samples_y]

print("I dataset (f14-f40):")
print(f"  ang_TI: min={min(f['ang_TI'] for f in feats_i):.1f}, max={max(f['ang_TI'] for f in feats_i):.1f}")
print(f"  ang_TP: min={min(f['ang_TP'] for f in feats_i):.1f}, max={max(f['ang_TP'] for f in feats_i):.1f}")
print(f"  d_TI:   min={min(f['d_TI'] for f in feats_i):.2f}, max={max(f['d_TI'] for f in feats_i):.2f}")
print(f"  d_TM:   min={min(f['d_TM'] for f in feats_i):.2f}, max={max(f['d_TM'] for f in feats_i):.2f}")
print(f"  d_TW:   min={min(f['d_TW'] for f in feats_i):.2f}, max={max(f['d_TW'] for f in feats_i):.2f}")

print("Y dataset (f16-f38):")
print(f"  ang_TI: min={min(f['ang_TI'] for f in feats_y):.1f}, max={max(f['ang_TI'] for f in feats_y):.1f}")
print(f"  ang_TP: min={min(f['ang_TP'] for f in feats_y):.1f}, max={max(f['ang_TP'] for f in feats_y):.1f}")
print(f"  d_TI:   min={min(f['d_TI'] for f in feats_y):.2f}, max={max(f['d_TI'] for f in feats_y):.2f}")
print(f"  d_TM:   min={min(f['d_TM'] for f in feats_y):.2f}, max={max(f['d_TM'] for f in feats_y):.2f}")
print(f"  d_TW:   min={min(f['d_TW'] for f in feats_y):.2f}, max={max(f['d_TW'] for f in feats_y):.2f}")

