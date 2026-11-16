"""
Tính toán biên an toàn (Safety Margins) chi tiết giữa Candidate 2 và các chữ cái đối kháng.
"""
import os
import sys
import glob
import json
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from backend.vision.alphabet_classifier import AlphabetClassifier
from scratch.task2_4_candidate_validation import load_fixture, load_pilot_jsonl

classifier = AlphabetClassifier()

# 1. Webcam X stats (stable hold)
pilot_path = "backend/data/diagnostics/pilot_pilot_20261108_184115.jsonl"
pilot_records = load_pilot_jsonl(pilot_path)
from collections import defaultdict, Counter
trials_dict = defaultdict(list)
for r in pilot_records:
    if r.get("expected_sign") == "X" and r.get("trial_id") is not None:
        trials_dict[r.get("trial_id")].append(r)
stable_pilot = []
for tid, f_list in trials_dict.items():
    n = len(f_list)
    for idx_in_trial, r in enumerate(f_list):
        if 3 <= idx_in_trial < n - 3:
            stable_pilot.append(r)

webcam_d_iw = [float(r["geometric"]["features"]["d_IW"]) for r in stable_pilot]
webcam_d_tw = [float(r["geometric"]["features"]["d_TW"]) for r in stable_pilot]
webcam_d_pw = [float(r["geometric"]["features"]["d_PW"]) for r in stable_pilot]
webcam_d_ti = [float(r["geometric"]["features"]["d_TI"]) for r in stable_pilot]

print("=== WEBCAM X (STABLE HOLD 228 FRAMES) ===")
print(f"d_IW: min={min(webcam_d_iw):.4f}, median={np.median(webcam_d_iw):.4f}, p95={np.percentile(webcam_d_iw, 95):.4f}, max={max(webcam_d_iw):.4f}")
print(f"d_TW: min={min(webcam_d_tw):.4f}, median={np.median(webcam_d_tw):.4f}, p95={np.percentile(webcam_d_tw, 95):.4f}, max={max(webcam_d_tw):.4f}")
print(f"d_PW: min={min(webcam_d_pw):.4f}, median={np.median(webcam_d_pw):.4f}, p95={np.percentile(webcam_d_pw, 95):.4f}, max={max(webcam_d_pw):.4f}")
print(f"d_TI: min={min(webcam_d_ti):.4f}, median={np.median(webcam_d_ti):.4f}, p95={np.percentile(webcam_d_ti, 95):.4f}, max={max(webcam_d_ti):.4f}")

# 2. Nearest non-X frames that satisfy ALL OTHER upright conditions except ONE
# For each parameter, we find the closest non-X frame
fixture_files = sorted(glob.glob("backend/data/alphabet_v2/static_landmarks/*.json"))
other_samples = []
for fpath in fixture_files:
    fname = os.path.basename(fpath).replace(".json", "")
    if fname == "X":
        continue
    fdata = load_fixture(fpath)
    for s in fdata.get("samples", []):
        lm = s.get("landmarks")
        if not lm or len(lm) != 21: continue
        feat = classifier.extract_features(lm)
        if feat:
            other_samples.append({"letter": fname, "feat": feat, "sample": s})

print(f"\nTotal other fixture frames tested: {len(other_samples)}")

# Test safety margins for Candidate 2:
# Cand 2 thresholds:
C2_IW = 1.55
C2_TW = 1.30
C2_PW = 0.65
C2_TI = 0.25

# Finding near-misses on other letters:
# Let's count how many conditions match for each other frame:
# Upright has 8 conditions:
# 1. d_TI <= C2_TI
# 2. d_IW <= C2_IW
# 3. i_curl >= 0.15
# 4. idx_straight <= 0.85
# 5. d_RW <= 0.70
# 6. d_PW <= C2_PW
# 7. 0.70 <= d_TW <= C2_TW
# 8. idx_tip_y <= -0.20

cond_match_counts = Counter()
letter_max_matches = defaultdict(int)
for item in other_samples:
    f = item["feat"]
    c1 = (f["d_TI"] <= C2_TI)
    c2 = (f["d_IW"] <= C2_IW)
    c3 = (f["index_curl"] >= 0.15)
    c4 = (f["idx_straight"] <= 0.85)
    c5 = (f["d_RW"] <= 0.70)
    c6 = (f["d_PW"] <= C2_PW)
    c7 = (0.70 <= f["d_TW"] <= C2_TW)
    c8 = (f["idx_tip_y"] <= -0.20)
    
    num_m = sum([c1, c2, c3, c4, c5, c6, c7, c8])
    cond_match_counts[num_m] += 1
    if num_m > letter_max_matches[item["letter"]]:
        letter_max_matches[item["letter"]] = num_m

print("\n=== DISTRIBUTION OF MATCHING CONDITIONS (OUT OF 8) FOR NON-X FRAMES ===")
for k in range(9):
    print(f"  {k}/8 conditions satisfied: {cond_match_counts[k]} frames")

print("\n=== MAX CONDITIONS SATISFIED PER LETTER ===")
for ltr, max_c in sorted(letter_max_matches.items()):
    print(f"  Letter {ltr}: max {max_c}/8 conditions satisfied")

# Detailed inspect frames with 7/8 conditions satisfied
print("\n=== INSPECTING FRAMES WITH 7/8 CONDITIONS SATISFIED (CRITICAL DEFENSE LINES) ===")
for item in other_samples:
    f = item["feat"]
    conds = {
        "d_TI<=0.25": (f["d_TI"] <= C2_TI),
        "d_IW<=1.55": (f["d_IW"] <= C2_IW),
        "i_curl>=0.15": (f["index_curl"] >= 0.15),
        "idx_straight<=0.85": (f["idx_straight"] <= 0.85),
        "d_RW<=0.70": (f["d_RW"] <= 0.70),
        "d_PW<=0.65": (f["d_PW"] <= C2_PW),
        "0.70<=d_TW<=1.30": (0.70 <= f["d_TW"] <= C2_TW),
        "idx_tip_y<=-0.20": (f["idx_tip_y"] <= -0.20),
    }
    if sum(conds.values()) >= 7:
        failed_cond = [k for k, v in conds.items() if not v]
        print(f"Letter {item['letter']} (fidx={item['sample'].get('frame_index')}): FAILED ONLY {failed_cond}")
        print(f"  Values: d_TI={f['d_TI']:.3f}, d_IW={f['d_IW']:.3f}, d_TW={f['d_TW']:.3f}, d_PW={f['d_PW']:.3f}, i_curl={f['index_curl']:.3f}, idx_str={f['idx_straight']:.3f}, d_RW={f['d_RW']:.3f}, tip_y={f['idx_tip_y']:.3f}")
