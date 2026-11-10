import json
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

with open("backend/data/vsl_alphabet_fixtures.json", "r", encoding="utf-8") as f:
    fixture = json.load(f)

print("Keys in fixture:", list(fixture.keys()))
samples = fixture.get("samples", {})
print("Sample letters in fixture:", list(samples.keys()))

c = VSLAlphabetClassifier()
if "R" in samples:
    r_data = samples["R"]
    print("R data keys:", list(r_data.keys()))
    lms = r_data.get("landmarks_21") or r_data.get("landmarks")
    feat = c.extract_features(lms)
    print("Fixture R features:")
    for k in ["d_IW", "d_MW", "d_RW", "d_PW", "d_IM", "idx_straight", "mid_straight", "idx_tip_y", "mid_tip_y", "idx_horiz_ratio", "idx_dy", "cross_indicator", "mcp_width_2d", "thumb_ring_dist"]:
        print(f"  {k}: {feat.get(k)}")
    diag = c.evaluate_rule_r_diagnostics(feat)
    print("Fixture R diag:", diag)

