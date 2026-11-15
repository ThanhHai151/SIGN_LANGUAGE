import json
import sys
sys.path.insert(0, ".")

from backend.vision.alphabet_classifier import alphabet_classifier

print("=== CHECKING RULE P ON D0509N ===")
with open("scratch/d0509n_extracted_raw.json", encoding="utf-8") as f:
    samples_p = json.load(f)

for s in samples_p[20:30]:
    lm = s.get("landmarks")
    if not lm or len(lm) < 21: continue
    diag = alphabet_classifier.evaluate_rule_p_diagnostics(lm)
    print(f"P f{s['frame_index']}: match={diag.get('matches')} failed_conds={diag.get('failed_conditions')}")

print("\n=== CHECKING RULE X ON D0517N ===")
with open("scratch/d0517n_extracted_raw.json", encoding="utf-8") as f:
    samples_x = json.load(f)

for s in samples_x[30:45]:
    lm = s.get("landmarks")
    if not lm or len(lm) < 21: continue
    diag = alphabet_classifier.evaluate_rule_x_diagnostics(lm)
    print(f"X f{s['frame_index']}: match={diag.get('matches')} failed_conds={diag.get('failed_conditions')}")

