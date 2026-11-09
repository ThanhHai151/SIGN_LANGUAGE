import sys
sys.path.insert(0, '.')
import json
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

clf = VSLAlphabetClassifier()

with open('backend/tests/fixtures/real_alphabet_landmarks.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

for letter in ['U', 'S']:
    print(f'=== Samples for {letter} ===')
    samples = [s for s in d['samples'] if s.get('letter_upper') == letter]
    for s in samples:
        f_idx = s.get('frame_idx')
        phase = s.get('phase')
        status = s.get('quality_status')
        lm = s.get('landmarks')
        if not lm:
            print(f'  f{f_idx}: no landmarks')
            continue
        feat = clf.extract_features(lm)
        res = clf.classify_detailed(lm)
        if letter == 'U':
            diag = clf.evaluate_rule_u_diagnostics(feat)
            failed = diag['failed_conditions']
        else:
            diag = clf.evaluate_rule_s_diagnostics(feat)
            failed = diag['failed_conditions']
        print(f"  f{f_idx} (phase={phase}, qual={status}): res={res.get('letter')}, status={res.get('status')}, failed_diag={failed}")
