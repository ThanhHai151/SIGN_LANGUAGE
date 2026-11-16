import sys
sys.path.insert(0, ".")
import json
from scratch.validate_new_rules import eval_rule_i_candidate, eval_rule_y_candidate
from backend.vision.alphabet_classifier import alphabet_classifier

fixture = json.load(open("backend/tests/fixtures/real_alphabet_landmarks.json", encoding="utf-8"))

i_samples = [s for s in fixture["samples"] if s["letter_upper"] == "I" and s.get("quality_status") == "ok"]
print(f"Fixture I samples: {len(i_samples)}")
for s in i_samples:
    feat = alphabet_classifier.extract_features(s["landmarks"])
    m_i = eval_rule_i_candidate(feat)
    m_y = eval_rule_y_candidate(feat)
    print(f"Frame {s['frame_idx']}: MATCH I = {m_i}, MATCH Y = {m_y}")

y_samples = [s for s in fixture["samples"] if s["letter_upper"] == "Y" and s.get("quality_status") == "ok"]
print(f"\nFixture Y samples: {len(y_samples)}")
for s in y_samples:
    feat = alphabet_classifier.extract_features(s["landmarks"])
    m_i = eval_rule_i_candidate(feat)
    m_y = eval_rule_y_candidate(feat)
    print(f"Frame {s['frame_idx']}: MATCH I = {m_i}, MATCH Y = {m_y}")

