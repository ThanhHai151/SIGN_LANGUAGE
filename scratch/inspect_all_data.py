import os
import json
import numpy as np

print("=== 1. Checking backend/data/vsl_dataset.npz ===")
ds1 = np.load('backend/data/vsl_dataset.npz', allow_pickle=True)
print("Keys:", ds1.files)
X1 = ds1['X']
y1 = ds1['y']
classes1 = ds1['classes']
print(f"X shape: {X1.shape}, y shape: {y1.shape}, classes len: {len(classes1)}")
for letter in ['T', 'U', 'S']:
    if letter in classes1:
        idx = list(classes1).index(letter)
        cnt = np.sum(y1 == idx)
        print(f"Class '{letter}': index={idx}, count={cnt}")
    else:
        print(f"Class '{letter}' NOT FOUND in classes1")

print("\n=== 2. Checking backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz ===")
ds2 = np.load('backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz', allow_pickle=True)
print("Keys:", ds2.files)
X2 = ds2['X']
y2 = ds2['y']
classes2 = ds2['classes']
landmarks2 = ds2['landmarks'] if 'landmarks' in ds2.files else None
print(f"X shape: {X2.shape}, y shape: {y2.shape}, classes len: {len(classes2)}, landmarks shape: {landmarks2.shape if landmarks2 is not None else None}")
for letter in ['T', 'U', 'S']:
    if letter in classes2:
        idx = list(classes2).index(letter)
        cnt = np.sum(y2 == idx)
        print(f"Class '{letter}': index={idx}, count={cnt}")
    else:
        print(f"Class '{letter}' NOT FOUND in classes2")

print("\n=== 3. Checking backend/data/alphabet_v2/static_landmarks/*.json ===")
for letter in ['T', 'U', 'S']:
    p = f'backend/data/alphabet_v2/static_landmarks/{letter}.json'
    if os.path.exists(p):
        with open(p, 'r', encoding='utf-8') as f:
            d = json.load(f)
            samples = d.get('samples', [])
            valid_samples = [s for s in samples if s.get('landmarks') is not None]
            phases = {}
            signers = set()
            for s in valid_samples:
                ph = s.get('phase', 'unknown')
                phases[ph] = phases.get(ph, 0) + 1
                if 'signer_id' in s:
                    signers.add(s['signer_id'])
            print(f"{letter}: total={len(samples)}, valid={len(valid_samples)}, phases={phases}, signers={signers}")
            if len(valid_samples) > 0:
                s0 = valid_samples[0]
                lms = s0['landmarks']
                print(f"   sample landmarks length: {len(lms)}, lm0 shape: {len(lms[0])}")
    else:
        print(f"{p} DOES NOT EXIST")

print("\n=== 4. Checking sign_descriptions.json for T, U, S ===")
with open('backend/data/sign_descriptions.json', 'r', encoding='utf-8') as f:
    desc = json.load(f)
for letter in ['T', 'U', 'S']:
    if letter in desc:
        print(f"Description of {letter}: {desc[letter]}")
    else:
        print(f"No description for {letter}")
