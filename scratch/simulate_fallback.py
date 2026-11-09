import json
import sys
sys.path.append('.')

from backend.vision.alphabet_classifier import AlphabetClassifier

classifier = AlphabetClassifier()

# Load records
with open('backend/data/diagnostics/pilot_pilot_20261107_2141.jsonl', 'r', encoding='utf-8') as f:
    records = [json.loads(line) for line in f if json.loads(line).get('trial_id')]

h_to_v_records = [r for r in records if r.get('expected_sign') == 'H' and r.get('final_sign') == 'V' and r.get('prediction_source') == 'geometric_rule']
k_to_v_records = [r for r in records if r.get('expected_sign') == 'K' and r.get('final_sign') == 'V' and r.get('prediction_source') == 'geometric_rule']

print(f"H->V count: {len(h_to_v_records)}")
print(f"K->V count: {len(k_to_v_records)}")

# Evaluate what rules match if Rule V is NOT included
fallback_matches_h = {}
for r in h_to_v_records:
    feat = r.get('geometric', {}).get('features', {})
    all_matches = classifier.evaluate_rules(feat)
    # Filter out 'V'
    non_v_matches = [m for m in all_matches if m[0] != 'V']
    if non_v_matches:
        match_keys = tuple(m[0] for m in non_v_matches)
    else:
        match_keys = 'NO_GEOMETRIC_RULE'
    fallback_matches_h[match_keys] = fallback_matches_h.get(match_keys, 0) + 1

print("\nH->V frames without Rule V - geometric rule match outcomes:")
for k, cnt in fallback_matches_h.items():
    print(f"  {k}: {cnt} frames")

fallback_matches_k = {}
for r in k_to_v_records:
    feat = r.get('geometric', {}).get('features', {})
    all_matches = classifier.evaluate_rules(feat)
    non_v_matches = [m for m in all_matches if m[0] != 'V']
    if non_v_matches:
        match_keys = tuple(m[0] for m in non_v_matches)
    else:
        match_keys = 'NO_GEOMETRIC_RULE'
    fallback_matches_k[match_keys] = fallback_matches_k.get(match_keys, 0) + 1

print("\nK->V frames without Rule V - geometric rule match outcomes:")
for k, cnt in fallback_matches_k.items():
    print(f"  {k}: {cnt} frames")
