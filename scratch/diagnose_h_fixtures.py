import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import json
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

classifier = VSLAlphabetClassifier()

for fixture_name in ['H', 'D0500N']:
    path = f'backend/data/alphabet_v2/static_landmarks/{fixture_name}.json'
    with open(path, 'r', encoding='utf-8') as f:
        samples = json.load(f).get('samples', [])
    
    print(f"\n=======================================================")
    print(f"DIAGNOSING {fixture_name} ({len(samples)} samples)")
    print(f"=======================================================")
    
    fail_reasons = {}
    h_horiz_fails = {}
    h_south_fails = {}
    h_webcam_fails = {}
    
    for s in samples:
        lms = s['landmarks']
        if not lms:
            continue
        pred, conf = classifier.classify(lms)
        if pred != 'H':
            feat = classifier.extract_features(lms)
            diag = classifier.evaluate_rule_h_diagnostics(feat)
            if diag:
                for k in diag.get('h_horiz', {}).get('failed_conditions', []):
                    h_horiz_fails[k] = h_horiz_fails.get(k, 0) + 1
                for k in diag.get('h_south', {}).get('failed_conditions', []):
                    h_south_fails[k] = h_south_fails.get(k, 0) + 1
                for k in diag.get('h_webcam', {}).get('failed_conditions', []):
                    h_webcam_fails[k] = h_webcam_fails.get(k, 0) + 1

    print(f"Failed H_horiz conditions: {sorted(h_horiz_fails.items(), key=lambda x: -x[1])[:8]}")
    print(f"Failed H_south conditions: {sorted(h_south_fails.items(), key=lambda x: -x[1])[:8]}")
    print(f"Failed H_webcam conditions: {sorted(h_webcam_fails.items(), key=lambda x: -x[1])[:8]}")
