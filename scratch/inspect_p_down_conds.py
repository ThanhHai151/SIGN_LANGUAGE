import json
import sys
sys.path.insert(0, ".")

from backend.vision.alphabet_classifier import alphabet_classifier

with open("scratch/d0509n_extracted_raw.json", encoding="utf-8") as f:
    samples_p = json.load(f)

for fidx in [15, 20, 25, 30]:
    lm = samples_p[fidx]["landmarks"]
    res = alphabet_classifier.classify_detailed(lm)
    feat = res.get("features", {})
    print(f"\n--- P Frame {samples_p[fidx]['frame_index']} ---")
    print(f"feat: d_IW={feat.get('d_IW'):.2f}, d_MW={feat.get('d_MW'):.2f}, d_RW={feat.get('d_RW'):.2f}, d_PW={feat.get('d_PW'):.2f}")
    print(f"      d_TI={feat.get('d_TI'):.2f}, d_TM={feat.get('d_TM'):.2f}, d_TW={feat.get('d_TW'):.2f}, ang_IM={feat.get('ang_IM_dir'):.1f}")
    print(f"      idx_tip_y={feat.get('idx_tip_y'):.2f}, mid_tip_y={feat.get('mid_tip_y'):.2f}, idx_dy={feat.get('idx_dy'):.2f}, idx_horiz={feat.get('idx_horiz_ratio'):.2f}")
    print(f"      idx_straight={feat.get('idx_straight'):.2f}, mid_straight={feat.get('mid_straight'):.2f}")

