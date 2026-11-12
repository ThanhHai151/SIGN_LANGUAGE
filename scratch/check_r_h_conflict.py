import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import json
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

c = VSLAlphabetClassifier()

# Check Canonical H
h_samples = json.load(open('backend/data/alphabet_v2/static_landmarks/H.json'))['samples']
h_conflicts = []
for s in h_samples:
    lms = s.get('landmarks')
    if not lms:
        continue
    feat = c.extract_features(lms)
    matches = c.evaluate_rules(feat)
    letters = [m[0] for m in matches]
    if len(letters) > 1:
        h_conflicts.append((s.get('frame_index'), letters))

print(f"Conflicts in Canonical H: {len(h_conflicts)} / {len(h_samples)}")
for fidx, letters in h_conflicts:
    print(f"  Frame {fidx}: {letters}")

# Check Canonical R
r_samples = json.load(open('backend/data/alphabet_v2/static_landmarks/R.json'))['samples']
r_conflicts = []
for s in r_samples:
    lms = s.get('landmarks')
    if not lms:
        continue
    feat = c.extract_features(lms)
    matches = c.evaluate_rules(feat)
    letters = [m[0] for m in matches]
    if len(letters) > 1:
        r_conflicts.append((s.get('frame_index'), letters))

print(f"\nConflicts in Canonical R: {len(r_conflicts)} / {len(r_samples)}")
for fidx, letters in r_conflicts:
    print(f"  Frame {fidx}: {letters}")

