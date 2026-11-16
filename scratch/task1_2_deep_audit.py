import json
import os
import numpy as np
from typing import Dict, Any, List, Optional
import math
import sys

sys.path.insert(0, os.path.abspath("."))
from backend.vision.alphabet_classifier import (
    alphabet_classifier,
    O_D_TI_MAX, O_D_TM_MAX, O_D_IW_MIN, O_D_MW_MIN,
    O_D_RW_MIN, O_D_PW_MIN, O_IDX_STRAIGHT_MAX, O_D_TW_MIN
)

def evaluate_frame_rule_o(lms_21):
    if lms_21 is None or len(lms_21) < 21:
        return None
    feat = alphabet_classifier.extract_features(lms_21)
    if feat is None:
        return None
    
    i_ext = feat.get("index_ext", False)
    m_ext = feat.get("middle_ext", False)
    r_ext = feat.get("ring_ext", False)
    p_ext = feat.get("pinky_ext", False)
    outer_gate = (not i_ext) and (not m_ext) and (not r_ext) and (not p_ext)

    d_ti = feat.get("d_TI", 0.0)
    d_tm = feat.get("d_TM", 0.0)
    d_iw = feat.get("d_IW", 0.0)
    d_mw = feat.get("d_MW", 0.0)
    d_rw = feat.get("d_RW", 0.0)
    d_pw = feat.get("d_PW", 0.0)
    idx_straight = feat.get("idx_straight", 0.0)
    d_tw = feat.get("d_TW", 0.0)

    conds = {
        "d_ti_le_max": d_ti <= O_D_TI_MAX,
        "d_tm_le_max": d_tm <= O_D_TM_MAX,
        "d_iw_ge_min": d_iw >= O_D_IW_MIN,
        "d_mw_ge_min": d_mw >= O_D_MW_MIN,
        "d_rw_ge_min": d_rw >= O_D_RW_MIN,
        "d_pw_ge_min": d_pw >= O_D_PW_MIN,
        "idx_straight_le_max": idx_straight <= O_IDX_STRAIGHT_MAX,
        "d_tw_ge_min": d_tw >= O_D_TW_MIN,
    }
    inner_all_pass = all(conds.values())
    rule_o_match = outer_gate and inner_all_pass

    detailed = alphabet_classifier.classify_detailed(lms_21)

    return {
        "feat": {
            "d_TI": d_ti,
            "d_TM": d_tm,
            "d_IW": d_iw,
            "d_MW": d_mw,
            "d_RW": d_rw,
            "d_PW": d_pw,
            "idx_straight": idx_straight,
            "d_TW": d_tw,
            "index_ext": i_ext,
            "middle_ext": m_ext,
            "ring_ext": r_ext,
            "pinky_ext": p_ext,
        },
        "outer_gate": outer_gate,
        "conds": conds,
        "inner_all_pass": inner_all_pass,
        "rule_o_match": rule_o_match,
        "classifier_status": detailed["status"],
        "classifier_letter": detailed["letter"],
        "classifier_conf": detailed["confidence"]
    }

def calc_stats(values: List[float]):
    if not values:
        return {}
    arr = np.array(values, dtype=float)
    return {
        "count": len(arr),
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
        "mean": round(float(np.mean(arr)), 4),
        "median": round(float(np.median(arr)), 4),
        "p5": round(float(np.percentile(arr, 5)), 4),
        "p25": round(float(np.percentile(arr, 25)), 4),
        "p75": round(float(np.percentile(arr, 75)), 4),
        "p95": round(float(np.percentile(arr, 95)), 4),
    }

def audit_fixture(letter: str):
    path = f"backend/data/alphabet_v2/static_landmarks/{letter}.json"
    if not os.path.exists(path):
        return {"exists": False}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    samples = data.get("samples", [])
    valid_frames = []
    has_nan_inf = 0
    missing_lms = 0
    phase_distribution = {}

    evaluated = []

    for item in samples:
        phase = item.get("phase", "UNKNOWN")
        phase_distribution[phase] = phase_distribution.get(phase, 0) + 1

        lms = item.get("landmarks")
        if not lms or len(lms) < 21:
            missing_lms += 1
            continue
        
        has_bad = False
        clean_lms = []
        for pt in lms:
            if not isinstance(pt, (list, tuple)) or len(pt) < 3:
                has_bad = True; break
            x, y, z = pt[0], pt[1], pt[2]
            if any(math.isnan(v) or math.isinf(v) for v in (x, y, z)):
                has_bad = True; break
            clean_lms.append([x, y, z])
        if has_bad:
            has_nan_inf += 1
            continue

        ev = evaluate_frame_rule_o(clean_lms)
        if ev:
            ev["frame_index"] = item.get("frame_index")
            ev["phase"] = phase
            ev["timestamp_sec"] = item.get("timestamp_sec")
            evaluated.append(ev)

    return {
        "exists": True,
        "path": path,
        "letter": data.get("letter"),
        "video_id": data.get("video_id"),
        "total_samples": len(samples),
        "missing_lms": missing_lms,
        "has_nan_inf": has_nan_inf,
        "valid_evaluable": len(evaluated),
        "phase_distribution": phase_distribution,
        "evaluated": evaluated
    }

def audit_all_and_save():
    fixtures = {let: audit_fixture(let) for let in ["O", "A", "S", "C"]}

    fixture_summary = {}
    for let, d in fixtures.items():
        ev_all = d.get("evaluated", [])
        ev_stable = [e for e in ev_all if e.get("phase") == "stable_pose"]
        ev_entry = [e for e in ev_all if e.get("phase") == "entry"]
        ev_exit = [e for e in ev_all if e.get("phase") == "exit"]

        def _summarize_set(ev_list):
            if not ev_list: return {}
            cond_keys = [
                "d_ti_le_max", "d_tm_le_max", "d_iw_ge_min", "d_mw_ge_min",
                "d_rw_ge_min", "d_pw_ge_min", "idx_straight_le_max", "d_tw_ge_min"
            ]
            return {
                "count": len(ev_list),
                "outer_gate_pass": sum(1 for e in ev_list if e["outer_gate"]),
                "outer_gate_pass_rate": round(sum(1 for e in ev_list if e["outer_gate"]) / len(ev_list), 4),
                "rule_o_match": sum(1 for e in ev_list if e["rule_o_match"]),
                "rule_o_match_rate": round(sum(1 for e in ev_list if e["rule_o_match"]) / len(ev_list), 4),
                "cond_passes": {c: sum(1 for e in ev_list if e["conds"].get(c)) for c in cond_keys},
                "cond_pass_rates": {c: round(sum(1 for e in ev_list if e["conds"].get(c)) / len(ev_list), 4) for c in cond_keys},
                "classifier_status_counts": {s: sum(1 for e in ev_list if e["classifier_status"] == s) for s in set(e["classifier_status"] for e in ev_list)},
                "classifier_letter_counts": {l: sum(1 for e in ev_list if e["classifier_letter"] == l) for l in set(e["classifier_letter"] for e in ev_list)},
                "features_stats": {
                    feat_name: calc_stats([e["feat"][feat_name] for e in ev_list])
                    for feat_name in ["d_TI", "d_TM", "d_IW", "d_MW", "d_RW", "d_PW", "idx_straight", "d_TW"]
                }
            }

        fixture_summary[let] = {
            "path": d["path"],
            "video_id": d["video_id"],
            "total_samples": d["total_samples"],
            "phase_distribution": d["phase_distribution"],
            "all_frames_summary": _summarize_set(ev_all),
            "stable_pose_summary": _summarize_set(ev_stable),
            "entry_summary": _summarize_set(ev_entry),
            "exit_summary": _summarize_set(ev_exit),
        }

    # Audit NPZ files
    npz_clean_path = "backend/data/alphabet_v2/datasets/vsl_alphabet_static_south_clean_v1.npz"
    npz_v2_path = "backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz"

    def _audit_npz(path):
        data = np.load(path, allow_pickle=True)
        classes = [str(c) for c in data["classes"]]
        y = data["y"]
        landmarks = data["landmarks"] # (N, 21, 3)
        res = {}
        for target in ["O", "A", "S", "C"]:
            if target not in classes: continue
            c_idx = classes.index(target)
            indices = np.where(y == c_idx)[0]
            ev_list = []
            for i in indices:
                ev = evaluate_frame_rule_o(landmarks[i].tolist())
                if ev: ev_list.append(ev)
            cond_keys = [
                "d_ti_le_max", "d_tm_le_max", "d_iw_ge_min", "d_mw_ge_min",
                "d_rw_ge_min", "d_pw_ge_min", "idx_straight_le_max", "d_tw_ge_min"
            ]
            res[target] = {
                "count": len(ev_list),
                "outer_gate_pass": sum(1 for e in ev_list if e["outer_gate"]),
                "outer_gate_pass_rate": round(sum(1 for e in ev_list if e["outer_gate"]) / len(ev_list), 4),
                "rule_o_match": sum(1 for e in ev_list if e["rule_o_match"]),
                "rule_o_match_rate": round(sum(1 for e in ev_list if e["rule_o_match"]) / len(ev_list), 4),
                "cond_passes": {c: sum(1 for e in ev_list if e["conds"].get(c)) for c in cond_keys},
                "cond_pass_rates": {c: round(sum(1 for e in ev_list if e["conds"].get(c)) / len(ev_list), 4) for c in cond_keys},
                "classifier_status_counts": {s: sum(1 for e in ev_list if e["classifier_status"] == s) for s in set(e["classifier_status"] for e in ev_list)},
                "classifier_letter_counts": {l: sum(1 for e in ev_list if e["classifier_letter"] == l) for l in set(e["classifier_letter"] for e in ev_list)},
                "features_stats": {
                    feat_name: calc_stats([e["feat"][feat_name] for e in ev_list])
                    for feat_name in ["d_TI", "d_TM", "d_IW", "d_MW", "d_RW", "d_PW", "idx_straight", "d_TW"]
                }
            }
        return res

    npz_clean_summary = _audit_npz(npz_clean_path)
    npz_v2_summary = _audit_npz(npz_v2_path)

    # Detailed audit of webcam pilot logs
    pilot_files = [
        "backend/data/diagnostics/pilot_pilot_20261108_193712_20261108_211742_1.jsonl",
        "backend/data/diagnostics/pilot_pilot_20261108_213800.jsonl",
        "backend/data/diagnostics/pilot_pilot_20261109_004312.jsonl",
        "backend/data/diagnostics/pilot_pilot_20261109_013304.jsonl",
        "backend/data/diagnostics/pilot_pilot_20261109_022547.jsonl"
    ]
    pilot_summaries = {}
    for pf in pilot_files:
        if not os.path.exists(pf): continue
        fname = os.path.basename(pf)
        trials = {}
        with open(pf, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                rec = json.loads(line)
                t_id = rec.get("trial_id")
                if t_id:
                    if t_id not in trials: trials[t_id] = []
                    trials[t_id].append(rec)
        t_stats = {}
        for t_id, f_list in trials.items():
            ts_list = [f.get("timestamp") for f in f_list if f.get("timestamp") is not None]
            dur = (ts_list[-1] - ts_list[0]) if len(ts_list) >= 2 else None
            final_signs = [f.get("final_sign") for f in f_list]
            rec_sources = [f.get("prediction_source") for f in f_list]
            gating_statuses = [f.get("gating_status") for f in f_list if f.get("gating_status") is not None]
            
            is_6s_protocol = dur is not None and abs(dur - 6.0) < 1.0
            t_stats[t_id] = {
                "frames_count": len(f_list),
                "duration_sec": round(dur, 3) if dur else None,
                "is_6s_protocol": is_6s_protocol,
                "final_sign_counts": {s: final_signs.count(s) for s in set(final_signs)},
                "source_counts": {s: rec_sources.count(s) for s in set(rec_sources)},
                "gating_status_counts": {s: gating_statuses.count(s) for s in set(gating_statuses)},
            }
        pilot_summaries[fname] = {
            "total_trials": len(trials),
            "trial_details": t_stats
        }

    full_report = {
        "fixtures": fixture_summary,
        "npz_clean_south_v1": npz_clean_summary,
        "npz_static_v2": npz_v2_summary,
        "pilot_summaries": pilot_summaries
    }

    out_path = "scratch/task1_2_full_audit_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    print(f"Full metrics written to {out_path}")

if __name__ == "__main__":
    audit_all_and_save()
