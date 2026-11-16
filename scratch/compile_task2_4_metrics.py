"""
Tập hợp toàn bộ số liệu định lượng cho TASK 2.4 phục vụ lập Báo cáo và cập nhật SSOT.
"""
import os
import sys
import glob
import json
from collections import defaultdict, Counter
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from backend.vision.alphabet_classifier import AlphabetClassifier

def load_pilot(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))
    return records

def load_fix(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

classifier = AlphabetClassifier()

# 1. Pilot X
pilot_path = "backend/data/diagnostics/pilot_pilot_20261108_184115.jsonl"
pilot_records = load_pilot(pilot_path)
trials_dict = defaultdict(list)
for r in pilot_records:
    if r.get("expected_sign") == "X" and r.get("trial_id") is not None:
        trials_dict[r.get("trial_id")].append(r)

active_pilot = []
stable_pilot = []
trans_pilot = []
for tid, f_list in trials_dict.items():
    n = len(f_list)
    for idx, r in enumerate(f_list):
        active_pilot.append(r)
        if idx < 3 or idx >= n - 3:
            r["_part"] = "transition"
            trans_pilot.append(r)
        else:
            r["_part"] = "stable_hold"
            stable_pilot.append(r)

# 2. Fixture X
fx_data = load_fix("backend/data/alphabet_v2/static_landmarks/X.json")
fx_valid = []
for s in fx_data.get("samples", []):
    lm = s.get("landmarks")
    if not lm or len(lm) != 21: continue
    feat = classifier.extract_features(lm)
    if feat:
        fx_valid.append({"sample": s, "feat": feat, "lm": lm})
fx_upright = [f for f in fx_valid if f["feat"]["idx_tip_y"] <= -0.20]
fx_forward = [f for f in fx_valid if f["feat"]["idx_tip_y"] >= 0.35]

# 3. Other Fixtures
other_fixtures = {}
for fpath in sorted(glob.glob("backend/data/alphabet_v2/static_landmarks/*.json")):
    fname = os.path.basename(fpath).replace(".json", "")
    if fname == "X": continue
    fdata = load_fix(fpath)
    val = []
    for s in fdata.get("samples", []):
        lm = s.get("landmarks")
        if not lm or len(lm) != 21: continue
        feat = classifier.extract_features(lm)
        if feat: val.append({"sample": s, "feat": feat, "lm": lm})
    other_fixtures[fname] = val

# 4. NPZ Dataset
npz_path = "backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz"
data_npz = np.load(npz_path, allow_pickle=True)
npz_classes = data_npz["classes"]
npz_y = data_npz["y"]
npz_lm = data_npz["landmarks"]

# Evaluator
def eval_cand(feat, ti_m, iw_m, tw_m, pw_m, min_str=None):
    if not feat: return False
    d_ti = feat.get("d_TI")
    d_iw = feat.get("d_IW")
    i_curl = feat.get("index_curl")
    idx_str = feat.get("idx_straight")
    d_rw = feat.get("d_RW")
    d_pw = feat.get("d_PW")
    d_tw = feat.get("d_TW")
    tip_y = feat.get("idx_tip_y")
    if any(v is None for v in [d_ti, d_iw, i_curl, idx_str, d_rw, d_pw, d_tw, tip_y]):
        return False
    is_up = (
        float(d_ti) <= ti_m
        and float(d_iw) <= iw_m
        and float(i_curl) >= 0.15
        and (min_str is None or float(idx_str) >= min_str)
        and float(idx_str) <= 0.85
        and float(d_rw) <= 0.70
        and float(d_pw) <= pw_m
        and 0.70 <= float(d_tw) <= tw_m
        and float(tip_y) <= -0.20
    )
    # Forward branch (untouched)
    d_mw = feat.get("d_MW")
    d_tm = feat.get("d_TM")
    idx_dy = feat.get("idx_dy")
    is_fwd = False
    if all(v is not None for v in [d_ti, d_iw, d_mw, d_tm, idx_dy, idx_str, d_rw, d_pw, d_tw, tip_y]):
        is_fwd = (
            0.35 <= float(d_ti) <= 0.60
            and 1.40 <= float(d_iw) <= 1.80
            and float(d_mw) >= 0.90
            and float(d_tm) >= 0.35
            and float(idx_dy) >= 0.85
            and 0.80 <= float(idx_str) <= 0.95
            and float(d_rw) <= 0.90
            and float(d_pw) <= 0.85
            and 1.15 <= float(d_tw) <= 1.45
            and 0.35 <= float(tip_y) <= 0.85
        )
    return is_up or is_fwd

candidates = {
    "Baseline": {"ti": 0.25, "iw": 1.10, "tw": 0.95, "pw": 0.60, "min_str": None},
    "Candidate 1": {"ti": 0.25, "iw": 1.50, "tw": 1.25, "pw": 0.65, "min_str": None},
    "Candidate 2": {"ti": 0.25, "iw": 1.55, "tw": 1.30, "pw": 0.65, "min_str": None},
    "Candidate 2B (Recommended)": {"ti": 0.25, "iw": 1.55, "tw": 1.30, "pw": 0.65, "min_str": 0.40},
    "Candidate 3": {"ti": 0.30, "iw": 1.55, "tw": 1.30, "pw": 0.70, "min_str": None},
    "Candidate 4": {"ti": 0.30, "iw": 1.60, "tw": 1.35, "pw": 0.75, "min_str": None},
}

summary = []
for cname, p in candidates.items():
    ti_m = p["ti"]
    iw_m = p["iw"]
    tw_m = p["tw"]
    pw_m = p["pw"]
    min_str = p["min_str"]

    # Pilot
    st_m = sum(1 for r in stable_pilot if eval_cand(r["geometric"]["features"], ti_m, iw_m, tw_m, pw_m, min_str))
    tr_m = sum(1 for r in trans_pilot if eval_cand(r["geometric"]["features"], ti_m, iw_m, tw_m, pw_m, min_str))
    act_m = sum(1 for r in active_pilot if eval_cand(r["geometric"]["features"], ti_m, iw_m, tw_m, pw_m, min_str))

    trial_accs = []
    for tid, f_list in trials_dict.items():
        st_in_t = [r for r in f_list if r.get("_part") == "stable_hold"]
        m_in_t = sum(1 for r in st_in_t if eval_cand(r["geometric"]["features"], ti_m, iw_m, tw_m, pw_m, min_str))
        trial_accs.append((m_in_t / len(st_in_t)) if st_in_t else 0.0)

    tr_ge_50 = sum(1 for a in trial_accs if a >= 0.50)
    tr_ge_70 = sum(1 for a in trial_accs if a >= 0.70)
    tr_ge_90 = sum(1 for a in trial_accs if a >= 0.90)

    # Fixture X
    fx_up_m = sum(1 for f in fx_upright if eval_cand(f["feat"], ti_m, iw_m, tw_m, pw_m, min_str))
    fx_fwd_m = sum(1 for f in fx_forward if eval_cand(f["feat"], ti_m, iw_m, tw_m, pw_m, min_str))
    fx_tot_m = sum(1 for f in fx_valid if eval_cand(f["feat"], ti_m, iw_m, tw_m, pw_m, min_str))

    # Fixture FPs
    fix_fps = {}
    tot_fix_fps = 0
    for ltr, flist in other_fixtures.items():
        m_count = sum(1 for f in flist if eval_cand(f["feat"], ti_m, iw_m, tw_m, pw_m, min_str))
        if m_count > 0:
            fix_fps[ltr] = m_count
            tot_fix_fps += m_count

    # NPZ FPs
    npz_fps = {}
    tot_npz_fps = 0
    for idx in range(len(npz_y)):
        true_l = npz_classes[npz_y[idx]]
        if true_l == "X": continue
        feat = classifier.extract_features(npz_lm[idx].tolist())
        if eval_cand(feat, ti_m, iw_m, tw_m, pw_m, min_str):
            npz_fps[true_l] = npz_fps.get(true_l, 0) + 1
            tot_npz_fps += 1

    summary.append({
        "candidate": cname,
        "params": p,
        "stable_m": st_m, "stable_tot": len(stable_pilot), "stable_pct": st_m / len(stable_pilot) * 100,
        "trans_m": tr_m, "trans_tot": len(trans_pilot), "trans_pct": tr_m / len(trans_pilot) * 100,
        "act_m": act_m, "act_tot": len(active_pilot), "act_pct": act_m / len(active_pilot) * 100,
        "tr_ge_70": tr_ge_70, "tr_ge_90": tr_ge_90,
        "fx_up_m": fx_up_m, "fx_up_tot": len(fx_upright), "fx_up_pct": fx_up_m / len(fx_upright) * 100,
        "fx_fwd_m": fx_fwd_m, "fx_fwd_tot": len(fx_forward), "fx_fwd_pct": fx_fwd_m / len(fx_forward) * 100,
        "fx_tot_m": fx_tot_m, "fx_tot_tot": len(fx_valid), "fx_tot_pct": fx_tot_m / len(fx_valid) * 100,
        "tot_fix_fps": tot_fix_fps, "fix_fps": fix_fps,
        "tot_npz_fps": tot_npz_fps, "npz_fps": npz_fps
    })

print(json.dumps(summary, indent=2))

