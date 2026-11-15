import json
import sys
sys.path.insert(0, ".")

import backend.vision.alphabet_classifier as ac

with open("scratch/d0517n_extracted_raw.json", encoding="utf-8") as f:
    samples_x = json.load(f)

for fidx in range(30, 65, 3):
    lm = samples_x[fidx]["landmarks"]
    res = ac.alphabet_classifier.classify_detailed(lm)
    feat = res.get("features", {})
    diag = ac.alphabet_classifier.evaluate_rule_x_diagnostics(feat)
    print(f"\n--- X Frame {samples_x[fidx]['frame_index']} ---")
    print(f"geom: status={res['status']}, letter={res['letter']}")
    print(f"diag: matched={diag.get('matched')} failed={diag.get('failed_conditions')}")
    print(f"feat: d_IW={feat.get('d_IW'):.2f}, d_MW={feat.get('d_MW'):.2f}, d_RW={feat.get('d_RW'):.2f}, d_PW={feat.get('d_PW'):.2f}")
    print(f"      d_TI={feat.get('d_TI'):.2f}, d_TM={feat.get('d_TM'):.2f}, d_TW={feat.get('d_TW'):.2f}")
    print(f"      idx_curl={feat.get('index_curl'):.2f}, idx_straight={feat.get('idx_straight'):.2f}, idx_tip_y={feat.get('idx_tip_y'):.2f}")
