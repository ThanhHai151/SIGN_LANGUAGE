import sys
sys.path.insert(0, ".")
import json
import glob
import numpy as np
import math

from backend.vision.alphabet_classifier import alphabet_classifier

# New rule evaluation function for testing
def eval_rule_i_candidate(feat):
    d_pw = feat.get("d_PW", 0.0)
    d_iw = feat.get("d_IW", 0.0)
    d_mw = feat.get("d_MW", 0.0)
    d_rw = feat.get("d_RW", 0.0)
    d_tw = feat.get("d_TW", 0.0)
    d_ti = feat.get("d_TI", 0.0)
    d_tm = feat.get("d_TM", 0.0)
    thumb_lat = feat.get("thumb_lat", 0.0)
    ang_tp = feat.get("ang_TP", 0.0)
    p_curl = feat.get("pinky_curl", 1.0)
    p_ext = feat.get("pinky_ext", False)
    i_ext = feat.get("index_ext", False)
    m_ext = feat.get("middle_ext", False)
    r_ext = feat.get("ring_ext", False)
    
    # 1. Pinky extended
    cond_pinky = (p_ext or p_curl <= 0.12) and (d_pw >= 1.35) and (p_curl <= 0.20)
    
    # 2. Three middle fingers curled
    cond_middle_curled = (not i_ext) and (not m_ext) and (not r_ext)
    cond_middle_dists = (d_iw <= 0.90) and (d_mw <= 0.80) and (d_rw <= 0.80)
    
    # 3. Pinky significantly longer than folded fingers
    max_curled = max(d_iw, d_mw, d_rw)
    cond_diff = (d_pw - max_curled) >= 0.55
    
    # 4. Thumb tucked against index/middle, NOT spread (NOT Shaka/Y)
    cond_thumb_tucked = (d_ti <= 0.45) and (d_tm <= 0.52) and (d_tw <= 1.15) and (thumb_lat <= 0.38 or ang_tp <= 33.0)
    
    return cond_pinky and cond_middle_curled and cond_middle_dists and cond_diff and cond_thumb_tucked

def eval_rule_y_candidate(feat):
    d_pw = feat.get("d_PW", 0.0)
    d_iw = feat.get("d_IW", 0.0)
    d_mw = feat.get("d_MW", 0.0)
    d_rw = feat.get("d_RW", 0.0)
    d_tw = feat.get("d_TW", 0.0)
    d_ti = feat.get("d_TI", 0.0)
    thumb_lat = feat.get("thumb_lat", 0.0)
    ang_tp = feat.get("ang_TP", 0.0)
    p_curl = feat.get("pinky_curl", 1.0)
    p_ext = feat.get("pinky_ext", False)
    i_ext = feat.get("index_ext", False)
    m_ext = feat.get("middle_ext", False)
    r_ext = feat.get("ring_ext", False)
    
    cond_pinky = (p_ext or p_curl <= 0.12) and (d_pw >= 1.30)
    cond_middle_curled = (not i_ext) and (not m_ext) and (not r_ext)
    cond_middle_dists = (d_iw <= 0.90) and (d_mw <= 0.80) and (d_rw <= 0.80)
    
    # Thumb spread outward into Shaka
    cond_thumb_spread = (d_tw >= 1.05) and (d_ti >= 0.40) and (thumb_lat >= 0.36 or ang_tp >= 33.0)
    
    return cond_pinky and cond_middle_curled and cond_middle_dists and cond_thumb_spread

# Run validation across all landmark files
all_files = glob.glob("backend/data/alphabet_v2/static_landmarks/*.json")
print("=== Testing Proposed Rule I and Y across all letters ===")
for f in sorted(all_files):
    data = json.load(open(f, encoding="utf-8"))
    let = data.get("letter")
    samples = [s for s in data.get("samples", []) if s.get("phase") == "stable_pose" and s.get("landmarks") is not None]
    
    i_matches = 0
    y_matches = 0
    total = len(samples)
    
    for s in samples:
        feat = alphabet_classifier.extract_features(s["landmarks"])
        if feat:
            if eval_rule_i_candidate(feat):
                i_matches += 1
            if eval_rule_y_candidate(feat):
                y_matches += 1
                
    if let == "I":
        print(f"Letter {let:2s}: {i_matches}/{total} MATCH I | {y_matches}/{total} MATCH Y")
    elif let == "Y":
        print(f"Letter {let:2s}: {i_matches}/{total} MATCH I | {y_matches}/{total} MATCH Y")
    else:
        if i_matches > 0 or y_matches > 0:
            print(f"Letter {let:2s}: FALSE POSITIVE! {i_matches} match I, {y_matches} match Y out of {total}")

print("=== Done checking ===")

