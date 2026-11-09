import json
import sys
sys.path.append('.')

from backend.vision.alphabet_classifier import AlphabetClassifier

classifier = AlphabetClassifier()

# Load the 140 H->V records and 1 K->V record
with open('backend/data/diagnostics/pilot_pilot_20261107_2141.jsonl', 'r', encoding='utf-8') as f:
    records = [json.loads(line) for line in f if json.loads(line).get('trial_id')]

h_to_v_records = [r for r in records if r.get('expected_sign') == 'H' and r.get('final_sign') == 'V' and r.get('prediction_source') == 'geometric_rule']
k_to_v_records = [r for r in records if r.get('expected_sign') == 'K' and r.get('final_sign') == 'V' and r.get('prediction_source') == 'geometric_rule']

print(f"H->V count: {len(h_to_v_records)}")
print(f"K->V count: {len(k_to_v_records)}")

# Let's see what other rules match or if other rules are even evaluated
# In AlphabetClassifier.classify_static_letter_features, let's see how matches are handled
# Let's inspect other rules in alphabet_classifier for i_ext and m_ext and not r_ext and not p_ext:
# What rules exist with this finger pattern?
# Let's run a test where Rule V condition fails (e.g. mid_horiz_ratio <= 0.50)

# Let's inspect the code of classify_static_letter_features
print("\nInspecting classification without Rule V:")
results_without_v = {}
for r in h_to_v_records:
    feat = r.get('geometric', {}).get('features', {})
    # Let's see what classify_static_letter_features returns currently:
    # If we evaluate all other rules:
    # We can inspect the matches list from classifier
    # Let's run classifier.classify_static_letter_features(feat)
    res = classifier.classify_static_letter_features(feat)
    # res is (sign, explanation, conf) or None
    # Let's see: if V is removed from matches, does any other rule match?
    # Let's mock classify without V
    pass


