"""
Task 2.2 Data Audit Script (READ-ONLY)
Analyzes:
1. pilot_pilot_20261108_184115.jsonl
2. Fixture X.json
3. Cross-label confusion on competing fixtures (P, S, O, E, C, A)
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath("."))

import json
import math
import hashlib
from collections import defaultdict, Counter
import numpy as np

PILOT_PATH = "backend/data/diagnostics/pilot_pilot_20261108_184115.jsonl"
FIXTURE_DIR = "backend/data/alphabet_v2/static_landmarks"

# Thresholds from alphabet_classifier.py
# Upright
X_D_TI_MAX = 0.25
X_D_IW_MAX = 1.10
X_IDX_CURL_MIN = 0.15
X_IDX_STRAIGHT_MAX = 0.85
X_D_RW_MAX = 0.70
X_D_PW_MAX = 0.60
X_D_TW_MIN = 0.70
X_D_TW_MAX = 0.95
X_IDX_TIP_Y_MAX = -0.20

# Forward
X_FWD_D_TI_MIN = 0.35
X_FWD_D_TI_MAX = 0.60
X_FWD_D_IW_MIN = 1.40
X_FWD_D_IW_MAX = 1.80
X_FWD_D_MW_MIN = 0.90
X_FWD_D_TM_MIN = 0.35
X_FWD_IDX_DY_MIN = 0.85
X_FWD_IDX_STRAIGHT_MIN = 0.80
X_FWD_IDX_STRAIGHT_MAX = 0.95
X_FWD_D_RW_MAX = 0.90
X_FWD_D_PW_MAX = 0.85
X_FWD_D_TW_MIN = 1.15
X_FWD_D_TW_MAX = 1.45
X_FWD_IDX_TIP_Y_MIN = 0.35
X_FWD_IDX_TIP_Y_MAX = 0.85

def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def eval_upright(feat):
    d_ti = feat.get("d_TI")
    d_iw = feat.get("d_IW")
    i_curl = feat.get("index_curl")
    idx_str = feat.get("idx_straight")
    d_rw = feat.get("d_RW")
    d_pw = feat.get("d_PW")
    d_tw = feat.get("d_TW")
    tip_y = feat.get("idx_tip_y")
    
    conds = {
        "d_ti_le_0.25": (d_ti <= X_D_TI_MAX) if d_ti is not None else None,
        "d_iw_le_1.10": (d_iw <= X_D_IW_MAX) if d_iw is not None else None,
        "i_curl_ge_0.15": (i_curl >= X_IDX_CURL_MIN) if i_curl is not None else None,
        "idx_straight_le_0.85": (idx_str <= X_IDX_STRAIGHT_MAX) if idx_str is not None else None,
        "d_rw_le_0.70": (d_rw <= X_D_RW_MAX) if d_rw is not None else None,
        "d_pw_le_0.60": (d_pw <= X_D_PW_MAX) if d_pw is not None else None,
        "d_tw_in_0.70_0.95": (X_D_TW_MIN <= d_tw <= X_D_TW_MAX) if d_tw is not None else None,
        "idx_tip_y_le_-0.20": (tip_y <= X_IDX_TIP_Y_MAX) if tip_y is not None else None,
    }
    matched = not any(v is False for v in conds.values()) if not any(v is None for v in conds.values()) else False
    return matched, conds

def eval_forward(feat):
    d_ti = feat.get("d_TI")
    d_iw = feat.get("d_IW")
    d_mw = feat.get("d_MW")
    d_tm = feat.get("d_TM")
    idx_dy = feat.get("idx_dy")
    idx_str = feat.get("idx_straight")
    d_rw = feat.get("d_RW")
    d_pw = feat.get("d_PW")
    d_tw = feat.get("d_TW")
    tip_y = feat.get("idx_tip_y")

    conds = {
        "d_ti_in_0.35_0.60": (X_FWD_D_TI_MIN <= d_ti <= X_FWD_D_TI_MAX) if d_ti is not None else None,
        "d_iw_in_1.40_1.80": (X_FWD_D_IW_MIN <= d_iw <= X_FWD_D_IW_MAX) if d_iw is not None else None,
        "d_mw_ge_0.90": (d_mw >= X_FWD_D_MW_MIN) if d_mw is not None else None,
        "d_tm_ge_0.35": (d_tm >= X_FWD_D_TM_MIN) if d_tm is not None else None,
        "idx_dy_ge_0.85": (idx_dy >= X_FWD_IDX_DY_MIN) if idx_dy is not None else None,
        "idx_straight_in_0.80_0.95": (X_FWD_IDX_STRAIGHT_MIN <= idx_str <= X_FWD_IDX_STRAIGHT_MAX) if idx_str is not None else None,
        "d_rw_le_0.90": (d_rw <= X_FWD_D_RW_MAX) if d_rw is not None else None,
        "d_pw_le_0.85": (d_pw <= X_FWD_D_PW_MAX) if d_pw is not None else None,
        "d_tw_in_1.15_1.45": (X_FWD_D_TW_MIN <= d_tw <= X_FWD_D_TW_MAX) if d_tw is not None else None,
        "idx_tip_y_in_0.35_0.85": (X_FWD_IDX_TIP_Y_MIN <= tip_y <= X_FWD_IDX_TIP_Y_MAX) if tip_y is not None else None,
    }
    matched = not any(v is False for v in conds.values()) if not any(v is None for v in conds.values()) else False
    return matched, conds

def main():
    print("=== TASK 2.2: PILOT FILE VERIFICATION & INTEGRITY ===")
    file_size = os.path.getsize(PILOT_PATH)
    file_hash = sha256_file(PILOT_PATH)
    print(f"Path: {PILOT_PATH}")
    print(f"Size: {file_size:,} bytes")
    print(f"SHA256: {file_hash}")

    total_lines = 0
    parse_errors = 0
    records = []
    
    with open(PILOT_PATH, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            total_lines += 1
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append(rec)
            except Exception as e:
                parse_errors += 1
                print(f"Parse error at line {idx}: {e}")

    print(f"Total lines: {total_lines}, Valid records: {len(records)}, Parse errors: {parse_errors}")

    # Check sessions & trials
    sessions = set()
    trials_dict = defaultdict(list)
    active_records = []
    idle_records = []

    for r in records:
        sid = r.get("session_id")
        tid = r.get("trial_id")
        exp = r.get("expected_sign")
        sessions.add(sid)
        if exp == "X" and tid is not None:
            active_records.append(r)
            trials_dict[tid].append(r)
        else:
            idle_records.append(r)

    print(f"\nUnique Sessions: {sessions}")
    print(f"Total Active (X) Frames: {len(active_records)}")
    print(f"Total Idle Frames: {len(idle_records)}")
    print(f"Total Trials: {len(trials_dict)}")

    # Check monotonicity & timestamps per trial
    trial_stats = []
    for tid, f_list in trials_dict.items():
        f_indices = [r.get("frame_idx") for r in f_list]
        ts_list = [r.get("timestamp") for r in f_list]
        is_fidx_mono = all(f_indices[i] <= f_indices[i+1] for i in range(len(f_indices)-1))
        is_ts_mono = all(ts_list[i] <= ts_list[i+1] for i in range(len(ts_list)-1))
        duration = ts_list[-1] - ts_list[0] if len(ts_list) > 1 else 0.0
        trial_stats.append({
            "trial_id": tid,
            "count": len(f_list),
            "fidx_mono": is_fidx_mono,
            "ts_mono": is_ts_mono,
            "duration": round(duration, 3),
            "start_ts": ts_list[0],
            "end_ts": ts_list[-1]
        })

    print("\n--- Trial Breakdown ---")
    for ts in trial_stats:
        print(f"Trial {ts['trial_id']}: {ts['count']} frames, duration: {ts['duration']}s, fidx_mono={ts['fidx_mono']}, ts_mono={ts['ts_mono']}")

    # Recognition performance in active frames
    print("\n=== RECOGNITION PERFORMANCE IN ACTIVE FRAMES ===")
    pred_counter = Counter([r.get("final_sign") for r in active_records])
    source_counter = Counter([r.get("prediction_source") for r in active_records])
    conf_counter = Counter([r.get("confirmed") for r in active_records])
    print(f"Predicted Signs: {pred_counter}")
    print(f"Prediction Sources: {source_counter}")
    print(f"Confirmed Count: {conf_counter}")

    # Check Static Head state
    static_head_status = Counter([str(r.get("static_head", {}).get("load_status")) for r in active_records])
    print(f"Static Head Load Status: {static_head_status}")

    # Independent Rule 18 evaluation
    print("\n=== INDEPENDENT EVALUATION OF RULE 18 (UPRIGHT & FORWARD) ===")
    upright_cond_pass = defaultdict(int)
    upright_cond_fail = defaultdict(int)
    upright_cond_none = defaultdict(int)
    upright_matches = 0

    fwd_cond_pass = defaultdict(int)
    fwd_cond_fail = defaultdict(int)
    fwd_cond_none = defaultdict(int)
    fwd_matches = 0

    joint_states = Counter()
    
    # Dead zone counters
    dead_d_ti = []
    dead_d_iw = []
    dead_idx_tip_y = []
    dead_d_tw = []

    # Features collector
    feat_values = defaultdict(list)

    for r in active_records:
        geom = r.get("geometric", {})
        feat = geom.get("features")
        if not feat:
            joint_states["no_features"] += 1
            continue

        for k in ["d_TI", "d_IW", "d_MW", "d_RW", "d_PW", "d_TW", "d_TM", "idx_straight", "idx_tip_y", "idx_dy", "index_curl", "palm_scale"]:
            if feat.get(k) is not None:
                feat_values[k].append(float(feat[k]))

        up_match, up_conds = eval_upright(feat)
        fwd_match, fwd_conds = eval_forward(feat)

        if up_match: upright_matches += 1
        if fwd_match: fwd_matches += 1

        for ck, cv in up_conds.items():
            if cv is True: upright_cond_pass[ck] += 1
            elif cv is False: upright_cond_fail[ck] += 1
            else: upright_cond_none[ck] += 1

        for ck, cv in fwd_conds.items():
            if cv is True: fwd_cond_pass[ck] += 1
            elif cv is False: fwd_cond_fail[ck] += 1
            else: fwd_cond_none[ck] += 1

        if up_match and not fwd_match:
            joint_states["upright_only"] += 1
        elif not up_match and fwd_match:
            joint_states["fwd_only"] += 1
        elif up_match and fwd_match:
            joint_states["both_match"] += 1
        else:
            joint_states["both_fail"] += 1

        # Check dead zones
        d_ti = feat.get("d_TI")
        d_iw = feat.get("d_IW")
        tip_y = feat.get("idx_tip_y")
        d_tw = feat.get("d_TW")

        if d_ti is not None and 0.25 < d_ti < 0.35:
            dead_d_ti.append((r.get("trial_id"), r.get("frame_idx"), d_ti))
        if d_iw is not None and 1.10 < d_iw < 1.40:
            dead_d_iw.append((r.get("trial_id"), r.get("frame_idx"), d_iw))
        if tip_y is not None and -0.20 < tip_y < 0.35:
            dead_idx_tip_y.append((r.get("trial_id"), r.get("frame_idx"), tip_y))
        if d_tw is not None and 0.95 < d_tw < 1.15:
            dead_d_tw.append((r.get("trial_id"), r.get("frame_idx"), d_tw))

    n_valid = len(active_records)
    print(f"\n--- Upright Hook ({n_valid} active frames) ---")
    print(f"Total Upright MATCH: {upright_matches} ({upright_matches/n_valid*100:.2f}%)")
    for ck in [
        "d_ti_le_0.25", "d_iw_le_1.10", "i_curl_ge_0.15", "idx_straight_le_0.85",
        "d_rw_le_0.70", "d_pw_le_0.60", "d_tw_in_0.70_0.95", "idx_tip_y_le_-0.20"
    ]:
        p = upright_cond_pass[ck]
        fl = upright_cond_fail[ck]
        pct = p / (p + fl) * 100 if (p + fl) > 0 else 0
        print(f"  {ck:<25}: PASS={p:<4} FAIL={fl:<4} ({pct:6.2f}%)")

    print(f"\n--- Forward Tilt ({n_valid} active frames) ---")
    print(f"Total Forward MATCH: {fwd_matches} ({fwd_matches/n_valid*100:.2f}%)")
    for ck in [
        "d_ti_in_0.35_0.60", "d_iw_in_1.40_1.80", "d_mw_ge_0.90", "d_tm_ge_0.35",
        "idx_dy_ge_0.85", "idx_straight_in_0.80_0.95", "d_rw_le_0.90", "d_pw_le_0.85",
        "d_tw_in_1.15_1.45", "idx_tip_y_in_0.35_0.85"
    ]:
        p = fwd_cond_pass[ck]
        fl = fwd_cond_fail[ck]
        pct = p / (p + fl) * 100 if (p + fl) > 0 else 0
        print(f"  {ck:<25}: PASS={p:<4} FAIL={fl:<4} ({pct:6.2f}%)")

    print(f"\n--- Joint State Distribution ---")
    for k, v in joint_states.items():
        print(f"  {k:<15}: {v} ({v/n_valid*100:.2f}%)")

    print(f"\n--- Geometric Dead-Zone Analysis ---")
    print(f"  d_TI in (0.25, 0.35)      : {len(dead_d_ti)} frames ({len(dead_d_ti)/n_valid*100:.2f}%)")
    print(f"  d_IW in (1.10, 1.40)      : {len(dead_d_iw)} frames ({len(dead_d_iw)/n_valid*100:.2f}%)")
    print(f"  idx_tip_y in (-0.20, 0.35): {len(dead_idx_tip_y)} frames ({len(dead_idx_tip_y)/n_valid*100:.2f}%)")
    print(f"  d_TW in (0.95, 1.15)      : {len(dead_d_tw)} frames ({len(dead_d_tw)/n_valid*100:.2f}%)")

    # Feature statistics
    print("\n--- Feature Distributions in Active Frames ---")
    for fk in ["index_curl", "idx_straight", "idx_tip_y", "idx_dy", "d_TI", "d_IW", "d_TW", "d_MW", "d_RW", "d_PW", "d_TM"]:
        vals = np.array(feat_values[fk])
        if len(vals) > 0:
            print(f"  {fk:<12}: min={vals.min():.4f}, p25={np.percentile(vals, 25):.4f}, median={np.median(vals):.4f}, mean={vals.mean():.4f}, p75={np.percentile(vals, 75):.4f}, max={vals.max():.4f}")

    # Kinematics analysis: Intra-trial breakdown
    # In each trial, identify entry, stable, exit
    print("\n=== KINEMATICS PARTITIONING (STABLE vs TRANSITION) ===")
    stable_frames = []
    transition_frames = []

    for tid, f_list in trials_dict.items():
        # First 3 frames and last 3 frames as transition boundaries
        n = len(f_list)
        for i, r in enumerate(f_list):
            if i < 3 or i >= n - 3:
                transition_frames.append(r)
            else:
                stable_frames.append(r)

    print(f"Partition: Stable Hold = {len(stable_frames)} frames, Transition = {len(transition_frames)} frames")

    # Performance on Stable Hold
    stable_preds = Counter([r.get("final_sign") for r in stable_frames])
    print(f"Stable Hold Predictions: {stable_preds}")

    # Fixture X.json Audit
    print("\n=== FIXTURE X.json AUDIT ===")
    x_fix_path = os.path.join(FIXTURE_DIR, "X.json")
    with open(x_fix_path, "r", encoding="utf-8") as f:
        x_fix = json.load(f)

    print(f"Letter: {x_fix.get('letter')}, Video ID: {x_fix.get('video_id')}, Valid frames: {x_fix.get('valid_frames')}, Total frames: {x_fix.get('total_frames')}")
    samples = x_fix.get("samples", [])
    print(f"Samples count: {len(samples)}")
    
    # Check fixture samples for Upright vs Forward
    from backend.vision.alphabet_classifier import alphabet_classifier
    fix_upright = 0
    fix_fwd = 0
    fix_pred = Counter()
    
    for s in samples:
        lms = s.get("landmarks")
        if not lms or len(lms) < 21: continue
        feat = alphabet_classifier.extract_features(lms)
        if not feat: continue
        um, _ = eval_upright(feat)
        fm, _ = eval_forward(feat)
        if um: fix_upright += 1
        if fm: fix_fwd += 1
        det = alphabet_classifier.classify_detailed(lms)
        fix_pred[det.get("letter")] += 1

    print(f"Fixture X Upright MATCH: {fix_upright}, Forward MATCH: {fix_fwd}")
    print(f"Fixture X Classifier predictions: {fix_pred}")

    # Cross-label confusion audit on competing fixtures
    print("\n=== CROSS-LABEL CONFUSION AUDIT (FIXTURES) ===")
    comp_letters = ["P", "S", "O", "E", "C", "A"]
    for let in comp_letters:
        l_path = os.path.join(FIXTURE_DIR, f"{let}.json")
        if not os.path.exists(l_path):
            print(f"Fixture {let}.json not found!")
            continue
        with open(l_path, "r", encoding="utf-8") as f:
            fdata = json.load(f)
        fsamples = fdata.get("samples", [])
        c_up = 0
        c_fwd = 0
        c_pred_x = 0
        valid_cnt = 0
        for s in fsamples:
            lms = s.get("landmarks")
            if not lms or len(lms) < 21: continue
            feat = alphabet_classifier.extract_features(lms)
            if not feat: continue
            valid_cnt += 1
            um, _ = eval_upright(feat)
            fm, _ = eval_forward(feat)
            if um: c_up += 1
            if fm: c_fwd += 1
            det = alphabet_classifier.classify_detailed(lms)
            if det.get("letter") == "X":
                c_pred_x += 1
        print(f"Fixture {let} ({valid_cnt} valid frames): Upright MATCH={c_up}, Forward MATCH={c_fwd}, Overall X Pred={c_pred_x}")

if __name__ == "__main__":
    main()
