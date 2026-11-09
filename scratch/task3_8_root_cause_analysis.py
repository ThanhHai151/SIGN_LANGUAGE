"""
TASK 3.8 - Comprehensive Root Cause Analysis Script
Computes:
1. Feature extraction formulas & sensitivity inspection.
2. Statistical distribution (min, p25, median, mean, p75, max, std) of idx_straight and d_TI:
   - By trial (rep01 to rep10 of pilot_20261109_004312)
   - Across sessions: Holdout 004312, Pilot 213800, Pilot 211742, NPZ O, Fixture O (D0506)
3. Contrast against non-O classes (B, P, X, A, C, S, T, E) on NPZ & fixtures.
4. Deep dive on B frame 54 of D0492.
5. Correlation between idx_straight and d_TI.
"""

import os
import sys
import json
import numpy as np
from collections import defaultdict

# Add workspace to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.vision.alphabet_classifier import AlphabetClassifier

def get_percentiles(arr):
    if len(arr) == 0:
        return {"N": 0, "min": None, "p25": None, "median": None, "mean": None, "p75": None, "max": None, "std": None}
    a = np.array(arr)
    return {
        "N": len(a),
        "min": round(float(np.min(a)), 4),
        "p25": round(float(np.percentile(a, 25)), 4),
        "median": round(float(np.median(a)), 4),
        "mean": round(float(np.mean(a)), 4),
        "p75": round(float(np.percentile(a, 75)), 4),
        "max": round(float(np.max(a)), 4),
        "std": round(float(np.std(a)), 4),
    }

def run_analysis():
    clf = AlphabetClassifier()
    out = {}

    # --- 1. SESSIONS COMPARISON ---
    # Session Holdout 004312
    p_004312 = "backend/data/diagnostics/pilot_pilot_20261109_004312.jsonl"
    recs_004312 = [json.loads(l) for l in open(p_004312, "r", encoding="utf-8")]
    act_004312 = [r for r in recs_004312 if r.get("expected_sign") == "O"]
    matched_004312 = [r for r in act_004312 if r.get("geometric", {}).get("letter") == "O"]
    unmatched_004312 = [r for r in act_004312 if r.get("geometric", {}).get("letter") != "O"]

    # Session 213800
    p_213800 = "backend/data/diagnostics/pilot_pilot_20261108_213800.jsonl"
    recs_213800 = [json.loads(l) for l in open(p_213800, "r", encoding="utf-8")]
    act_213800 = [r for r in recs_213800 if r.get("trial_id") and r.get("geometric", {}).get("features")]

    # Session 211742
    p_211742 = "backend/data/diagnostics/pilot_pilot_20261108_193712_20261108_211742_1.jsonl"
    recs_211742 = [json.loads(l) for l in open(p_211742, "r", encoding="utf-8")]
    act_211742 = [r for r in recs_211742 if r.get("trial_id") and r.get("geometric", {}).get("features")]

    # NPZ O
    npz_path = "backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz"
    npz = np.load(npz_path, allow_pickle=True)
    classes = list(npz["classes"])
    y = npz["y"]
    lms = npz["landmarks"]
    meta = json.loads(str(npz["metadata_json"]))

    o_class_idx = classes.index("O")
    o_npz_lms = [lms[i].tolist() for i in range(len(y)) if y[i] == o_class_idx]
    o_npz_feats = [clf.extract_features(lm) for lm in o_npz_lms if clf.extract_features(lm) is not None]

    # Fixture O (D0506)
    fix_o_path = "backend/data/alphabet_v2/static_landmarks/O.json"
    fix_o_data = json.load(open(fix_o_path, "r", encoding="utf-8"))
    fix_o_samples = [s for s in fix_o_data.get("samples", []) if s.get("landmarks")]
    fix_o_feats = [clf.extract_features(s["landmarks"]) for s in fix_o_samples if clf.extract_features(s["landmarks"]) is not None]
    fix_o_hold_feats = [clf.extract_features(s["landmarks"]) for s in fix_o_samples if s.get("phase") == "stable_pose" and clf.extract_features(s["landmarks"]) is not None]

    sessions_stats = {
        "holdout_004312_all_active": {
            "idx_straight": get_percentiles([r["geometric"]["features"]["idx_straight"] for r in act_004312]),
            "d_TI": get_percentiles([r["geometric"]["features"]["d_TI"] for r in act_004312]),
        },
        "holdout_004312_matched": {
            "idx_straight": get_percentiles([r["geometric"]["features"]["idx_straight"] for r in matched_004312]),
            "d_TI": get_percentiles([r["geometric"]["features"]["d_TI"] for r in matched_004312]),
        },
        "holdout_004312_unmatched": {
            "idx_straight": get_percentiles([r["geometric"]["features"]["idx_straight"] for r in unmatched_004312]),
            "d_TI": get_percentiles([r["geometric"]["features"]["d_TI"] for r in unmatched_004312]),
        },
        "pilot_213800_active": {
            "idx_straight": get_percentiles([r["geometric"]["features"]["idx_straight"] for r in act_213800]),
            "d_TI": get_percentiles([r["geometric"]["features"]["d_TI"] for r in act_213800]),
        },
        "pilot_211742_active": {
            "idx_straight": get_percentiles([r["geometric"]["features"]["idx_straight"] for r in act_211742]),
            "d_TI": get_percentiles([r["geometric"]["features"]["d_TI"] for r in act_211742]),
        },
        "npz_o_samples": {
            "idx_straight": get_percentiles([f["idx_straight"] for f in o_npz_feats]),
            "d_TI": get_percentiles([f["d_TI"] for f in o_npz_feats]),
        },
        "fixture_o_all": {
            "idx_straight": get_percentiles([f["idx_straight"] for f in fix_o_feats]),
            "d_TI": get_percentiles([f["d_TI"] for f in fix_o_feats]),
        },
        "fixture_o_hold": {
            "idx_straight": get_percentiles([f["idx_straight"] for f in fix_o_hold_feats]),
            "d_TI": get_percentiles([f["d_TI"] for f in fix_o_hold_feats]),
        }
    }
    out["sessions_stats"] = sessions_stats

    # --- 2. TRIAL-BY-TRIAL STATS FOR HOLDOUT 004312 ---
    trials_004312 = defaultdict(list)
    for r in act_004312:
        trials_004312[r.get("trial_id")].append(r)

    trials_stats = {}
    for tid in sorted(trials_004312.keys()):
        recs = trials_004312[tid]
        idx_s = [r["geometric"]["features"]["idx_straight"] for r in recs]
        dtis = [r["geometric"]["features"]["d_TI"] for r in recs]
        m_cnt = sum(1 for r in recs if r.get("geometric", {}).get("letter") == "O")
        trials_stats[tid[-5:]] = {
            "total_frames": len(recs),
            "matched_frames": m_cnt,
            "match_rate": round(m_cnt / len(recs), 4),
            "idx_straight": get_percentiles(idx_s),
            "d_TI": get_percentiles(dtis)
        }
    out["trials_stats_004312"] = trials_stats

    # --- 3. CONTRAST AGAINST NON-O CLASSES ---
    target_contrast_classes = ["B", "P", "X", "A", "C", "S", "T", "E"]
    contrast_stats = {}

    for cls_name in target_contrast_classes:
        if cls_name in classes:
            c_idx = classes.index(cls_name)
            c_lms = [lms[i].tolist() for i in range(len(y)) if y[i] == c_idx]
            c_feats = [clf.extract_features(lm) for lm in c_lms if clf.extract_features(lm) is not None]
            contrast_stats[f"NPZ_{cls_name}"] = {
                "idx_straight": get_percentiles([f["idx_straight"] for f in c_feats]),
                "d_TI": get_percentiles([f["d_TI"] for f in c_feats]),
                "d_TM": get_percentiles([f["d_TM"] for f in c_feats]),
                "d_TW": get_percentiles([f["d_TW"] for f in c_feats]),
            }

    out["contrast_classes_npz"] = contrast_stats

    # --- 4. B FRAME 54 OF D0492 ANALYSIS ---
    # Find sample 39 in NPZ
    m39 = meta[39]
    lm39 = lms[39].tolist()
    feat39 = clf.extract_features(lm39)
    diag39 = clf.evaluate_rule_o_diagnostics(feat39)

    out["b_frame_54"] = {
        "npz_index": 39,
        "video_id": m39.get("video_id"),
        "frame_index": m39.get("frame_index"),
        "timestamp_sec": m39.get("timestamp_sec"),
        "true_label": classes[y[39]],
        "features": {
            "idx_straight": round(feat39["idx_straight"], 4),
            "d_TI": round(feat39["d_TI"], 4),
            "d_TM": round(feat39["d_TM"], 4),
            "d_TW": round(feat39["d_TW"], 4),
            "d_IW": round(feat39["d_IW"], 4),
            "d_MW": round(feat39["d_MW"], 4),
            "d_RW": round(feat39["d_RW"], 4),
            "d_PW": round(feat39["d_PW"], 4),
            "index_ext": feat39["index_ext"],
            "middle_ext": feat39["middle_ext"],
            "ring_ext": feat39["ring_ext"],
            "pinky_ext": feat39["pinky_ext"]
        },
        "rule_o_evaluation": diag39
    }

    # Save artifact
    out_path = "scratch/task3_8_root_cause_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Summary written to {out_path}")

if __name__ == "__main__":
    run_analysis()
