import json
import sys
import numpy as np

sys.path.insert(0, ".")
from backend.vision.alphabet_classifier import AlphabetClassifier
from backend.vision.feature_extractor import extract_dual_hand_features, calculate_distance

clf = AlphabetClassifier()
data = json.load(open("backend/data/alphabet_v2/static_landmarks/O.json", "r", encoding="utf-8"))
samples = [s for s in data["samples"] if s.get("landmarks")]

print(f"Total valid samples in O.json: {len(samples)}")

hold_indices = [22, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46]

rows = []
for i, s in enumerate(samples):
    f_idx = s["frame_index"]
    ts = s.get("timestamp_sec", 0.0)
    lm = s["landmarks"]
    
    # Feature extraction
    feats = clf.extract_features(lm)
    _, meta = extract_dual_hand_features(lm, None)
    states = meta.get("primary_states", {})
    
    # Finger extended / upward
    # TIP.y < PIP.y means finger points upward
    idx_upward = bool(states.get("index_extended", False))
    mid_upward = bool(states.get("middle_extended", False))
    ring_upward = bool(states.get("ring_extended", False))
    pinky_upward = bool(states.get("pinky_extended", False))
    
    # Outer gate in rule 6: not i_ext and not m_ext and not r_ext and not p_ext
    outer_gate_pass = (not idx_upward and not mid_upward and not ring_upward and not pinky_upward)
    
    # Palm normal calculation (facing camera: normal_z)
    # MCP5 = lm[5] (index MCP), MCP17 = lm[17] (pinky MCP), Wrist = lm[0]
    p0 = np.array(lm[0])
    p5 = np.array(lm[5])
    p17 = np.array(lm[17])
    v1 = p5 - p0
    v2 = p17 - p0
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    normal_unit = normal / norm if norm > 1e-7 else np.array([0, 0, 0])
    # normal_unit[2] > 0 or < 0?
    palm_facing_camera = bool(normal_unit[2] < 0) # in standard camera coordinates, -z points towards camera
    
    idx_straight = feats["idx_straight"]
    d_ti = feats["d_TI"]
    d_tm = feats["d_TM"]
    d_tw = feats["d_TW"]
    d_iw = feats["d_IW"]
    d_mw = feats["d_MW"]
    d_rw = feats["d_RW"]
    d_pw = feats["d_PW"]
    
    # Rule 6 evaluation
    # O_D_TI_MAX = 0.22, O_D_TM_MAX = 0.35, O_D_IW_MIN = 0.95, O_D_MW_MIN = 0.90,
    # O_D_RW_MIN = 0.80, O_D_PW_MIN = 0.75, O_IDX_STRAIGHT_MAX = 0.48, O_D_TW_MIN = 0.85
    r6_conditions = {
        "outer_gate": outer_gate_pass,
        "d_ti_le_0.22": d_ti <= 0.22,
        "d_tm_le_0.35": d_tm <= 0.35,
        "d_iw_ge_0.95": d_iw >= 0.95,
        "d_mw_ge_0.90": d_mw >= 0.90,
        "d_rw_ge_0.80": d_rw >= 0.80,
        "d_pw_ge_0.75": d_pw >= 0.75,
        "idx_straight_le_0.48": idx_straight <= 0.48,
        "d_tw_ge_0.85": d_tw >= 0.85,
    }
    r6_pass = all(r6_conditions.values())
    
    # Also check full classifier
    matches = clf.evaluate_rules(feats)
    o_in_matches = any(m[0] == "O" for m in matches)
    classified_letter, conf = clf.classify(lm)
    
    rows.append({
        "sample_idx": i,
        "frame_index": f_idx,
        "timestamp_sec": ts,
        "is_hold": f_idx in hold_indices,
        "idx_straight": idx_straight,
        "d_TI": d_ti,
        "d_TM": d_tm,
        "d_TW": d_tw,
        "d_IW": d_iw,
        "d_MW": d_mw,
        "d_RW": d_rw,
        "d_PW": d_pw,
        "wrist_dist": feats["wrist_dist"] if "wrist_dist" in feats else d_iw, # or d_iw
        "palm_normal_z": float(normal_unit[2]),
        "palm_facing_camera": palm_facing_camera,
        "idx_upward": idx_upward,
        "mid_upward": mid_upward,
        "ring_upward": ring_upward,
        "pinky_upward": pinky_upward,
        "outer_gate_pass": outer_gate_pass,
        "r6_pass": r6_pass,
        "o_in_matches": o_in_matches,
        "classified_letter": classified_letter,
        "confidence": conf,
        "failed_r6_conds": [k for k, v in r6_conditions.items() if not v]
    })

print("\n=== 12 STABLE HOLD FRAMES (D0506) ===")
hold_rows = [r for r in rows if r["is_hold"]]
print(f"Count: {len(hold_rows)}")
for r in hold_rows:
    print(f"frame {r['frame_index']:02d} ({r['timestamp_sec']:.3f}s): idx_str={r['idx_straight']:.4f}, d_TI={r['d_TI']:.4f}, palm_facing={r['palm_facing_camera']} (nz={r['palm_normal_z']:.2f}), up=[{r['idx_upward']}, {r['mid_upward']}, {r['ring_upward']}, {r['pinky_upward']}], d_IW={r['d_IW']:.2f}, R6={r['r6_pass']} (pred={r['classified_letter']}, conf={r['confidence']})")

print("\n=== 8 NON-HOLD FRAMES (D0506) ===")
non_hold_rows = [r for r in rows if not r["is_hold"]]
print(f"Count: {len(non_hold_rows)}")
for r in non_hold_rows:
    print(f"frame {r['frame_index']:02d} ({r['timestamp_sec']:.3f}s): idx_str={r['idx_straight']:.4f}, d_TI={r['d_TI']:.4f}, palm_facing={r['palm_facing_camera']} (nz={r['palm_normal_z']:.2f}), up=[{r['idx_upward']}, {r['mid_upward']}, {r['ring_upward']}, {r['pinky_upward']}], R6={r['r6_pass']}, failed={r['failed_r6_conds']}")

print("\n=== COMPARISON OF STATS ===")
for name, subset in [("12 Hold Frames", hold_rows), ("8 Non-Hold Frames", non_hold_rows), ("All 20 Frames", rows)]:
    idx_s = [r["idx_straight"] for r in subset]
    d_ti = [r["d_TI"] for r in subset]
    print(f"--- {name} (N={len(subset)}) ---")
    print(f"  idx_straight: min={min(idx_s):.4f}, med={np.median(idx_s):.4f}, mean={np.mean(idx_s):.4f}, max={max(idx_s):.4f}, std={np.std(idx_s):.4f}")
    print(f"  d_TI:         min={min(d_ti):.4f}, med={np.median(d_ti):.4f}, mean={np.mean(d_ti):.4f}, max={max(d_ti):.4f}, std={np.std(d_ti):.4f}")

