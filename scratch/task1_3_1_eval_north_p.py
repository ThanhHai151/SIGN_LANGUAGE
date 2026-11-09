import sys
sys.path.insert(0, '.')
import json
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

clf = VSLAlphabetClassifier()

# 1. Evaluate on P.json
with open('backend/data/alphabet_v2/static_landmarks/P.json', 'r', encoding='utf-8') as f:
    p_data = json.load(f)

samples = p_data.get('samples', [])
print(f'=== Evaluating North P Rule on P.json (Total samples: {len(samples)}) ===')

matched_frames = []
missed_frames = []

for s in samples:
    f_idx = s.get('frame_index') or s.get('frame_idx')
    phase = s.get('phase')
    lms = s.get('landmarks')
    if not lms:
        missed_frames.append((f_idx, phase, 'NO_LANDMARKS', []))
        continue
    feat = clf.extract_features(lms)
    diag = clf.evaluate_rule_p_diagnostics(feat)
    if diag and diag['matched']:
        matched_frames.append((f_idx, phase))
    else:
        failed = diag['failed_conditions'] if diag else ['FEAT_NONE']
        missed_frames.append((f_idx, phase, 'FAILED_CONDS', failed))

print(f"Matched: {len(matched_frames)} / {len(samples)} ({len(matched_frames)/len(samples)*100:.1f}%)")
print(f"Missed:  {len(missed_frames)} / {len(samples)} ({len(missed_frames)/len(samples)*100:.1f}%)")

print("\nMatched frames:")
for f_idx, phase in matched_frames:
    print(f"  Frame {f_idx:2d} (phase={phase})")

print("\nMissed frames details:")
for f_idx, phase, reason, failed in missed_frames:
    idx_str = str(f_idx) if f_idx is not None else "None"
    print(f"  Frame {idx_str:4s} (phase={phase}): reason={reason}, failed={failed}")

# 2. Check existing tests in test_alphabet_classifier.py
print("\n=== Checking existing tests in test_alphabet_classifier.py ===")
with open('backend/tests/test_alphabet_classifier.py', 'r', encoding='utf-8') as f:
    test_src = f.read()

for tname in ['test_letter_p', 'test_p_matches_on_d0509b_fixtures', 'test_p_and_k_strict_separation']:
    pos = test_src.find(f'def {tname}')
    if pos != -1:
        end = test_src.find('\n    def ', pos + 1)
        if end == -1: end = pos + 1000
        print(f"\n--- {tname} ---")
        print(test_src[pos:end].strip())
