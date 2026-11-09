"""
Script kiểm định độc lập và đánh giá định lượng các phương án điều chỉnh Rule 18 (X)
Phục vụ TASK 2.4 - THIẾT KẾ PHƯƠNG ÁN & KIỂM ĐỊNH ĐỘC LẬP CHỮ X (READ-ONLY)
"""

import os
import sys
import json
import glob
from collections import defaultdict, Counter
import numpy as np

# Đảm bảo import được backend
sys.path.insert(0, os.path.abspath("."))
from backend.vision.alphabet_classifier import AlphabetClassifier

def load_pilot_jsonl(filepath):
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records

def load_fixture(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

# Hàm kiểm tra điều kiện Upright độc lập với các ngưỡng tùy biến
def eval_x_upright_custom(feat, d_ti_max, d_iw_max, d_tw_max, d_pw_max,
                         idx_curl_min=0.15, idx_straight_max=0.85,
                         d_rw_max=0.70, d_tw_min=0.70, idx_tip_y_max=-0.20):
    if not feat or not isinstance(feat, dict):
        return False
    d_ti = feat.get("d_TI")
    d_iw = feat.get("d_IW")
    i_curl = feat.get("index_curl")
    idx_straight = feat.get("idx_straight")
    d_rw = feat.get("d_RW")
    d_pw = feat.get("d_PW")
    d_tw = feat.get("d_TW")
    idx_tip_y = feat.get("idx_tip_y")

    if any(v is None for v in [d_ti, d_iw, i_curl, idx_straight, d_rw, d_pw, d_tw, idx_tip_y]):
        return False

    return (
        float(d_ti) <= d_ti_max
        and float(d_iw) <= d_iw_max
        and float(i_curl) >= idx_curl_min
        and float(idx_straight) <= idx_straight_max
        and float(d_rw) <= d_rw_max
        and float(d_pw) <= d_pw_max
        and d_tw_min <= float(d_tw) <= d_tw_max
        and float(idx_tip_y) <= idx_tip_y_max
    )

# Hàm kiểm tra Forward Tilt (giữ nguyên production 100%)
def eval_x_forward(feat):
    if not feat or not isinstance(feat, dict):
        return False
    d_ti = feat.get("d_TI")
    d_iw = feat.get("d_IW")
    d_mw = feat.get("d_MW")
    d_tm = feat.get("d_TM")
    idx_dy = feat.get("idx_dy")
    idx_straight = feat.get("idx_straight")
    d_rw = feat.get("d_RW")
    d_pw = feat.get("d_PW")
    d_tw = feat.get("d_TW")
    idx_tip_y = feat.get("idx_tip_y")

    if any(v is None for v in [d_ti, d_iw, d_mw, d_tm, idx_dy, idx_straight, d_rw, d_pw, d_tw, idx_tip_y]):
        return False

    return (
        0.35 <= float(d_ti) <= 0.60
        and 1.40 <= float(d_iw) <= 1.80
        and float(d_mw) >= 0.90
        and float(d_tm) >= 0.35
        and float(idx_dy) >= 0.85
        and 0.80 <= float(idx_straight) <= 0.95
        and float(d_rw) <= 0.90
        and float(d_pw) <= 0.85
        and 1.15 <= float(d_tw) <= 1.45
        and 0.35 <= float(idx_tip_y) <= 0.85
    )

def eval_x_match_custom(feat, d_ti_max, d_iw_max, d_tw_max, d_pw_max):
    return eval_x_upright_custom(feat, d_ti_max, d_iw_max, d_tw_max, d_pw_max) or eval_x_forward(feat)

def simulate_pipeline(classifier, landmarks, feat, d_ti_max, d_iw_max, d_tw_max, d_pw_max):
    """
    Mô phỏng chính xác classify_detailed() nhưng thay thế Rule 18 bằng phiên bản tùy biến.
    Tất cả các rules 1-17 giữ nguyên.
    """
    if feat is None:
        if landmarks is None or len(landmarks) != 21:
            return {"letter": "UNKNOWN", "status": "INVALID_INPUT", "matches": []}
        feat = classifier.extract_features(landmarks)
    if feat is None:
        return {"letter": "UNKNOWN", "status": "INVALID_INPUT", "matches": []}

    # Đánh giá tất cả rules 1-17 bằng cách truyền dummy feature khiến Rule 18 gốc không match
    dummy_feat = dict(feat)
    dummy_feat["d_TI"] = 999.0
    dummy_feat["idx_dy"] = -999.0
    base_matches = classifier.evaluate_rules(dummy_feat)

    # Bây giờ đánh giá Rule 18 phiên bản custom
    is_x_up = eval_x_upright_custom(feat, d_ti_max, d_iw_max, d_tw_max, d_pw_max)
    is_x_fwd = eval_x_forward(feat)

    sim_matches = list(base_matches)
    if is_x_up or is_x_fwd:
        # Giữ nguyên logic production: loại bỏ A, S, O, E, C nếu có
        sim_matches = [m for m in sim_matches if m[0] not in ("A", "S", "O", "E", "C")]
        desc = "dựng đứng" if is_x_up else "chúi trước"
        sim_matches.append(("X", f"Chữ X ({desc})", 0.95))

    if len(sim_matches) == 1:
        return {"letter": sim_matches[0][0], "status": "MATCH", "matches": sim_matches}
    elif len(sim_matches) > 1:
        return {"letter": "UNKNOWN", "status": "CONFLICT", "matches": sim_matches}
    else:
        return {"letter": "UNKNOWN", "status": "NO_MATCH", "matches": []}

def main():
    classifier = AlphabetClassifier()

    # 1. Load dữ liệu pilot X
    pilot_path = "backend/data/diagnostics/pilot_pilot_20261108_184115.jsonl"
    pilot_records = load_pilot_jsonl(pilot_path)
    
    trials_dict = defaultdict(list)
    for r in pilot_records:
        if r.get("expected_sign") == "X" and r.get("trial_id") is not None:
            trials_dict[r.get("trial_id")].append(r)
            
    active_pilot = []
    stable_pilot = []
    trans_pilot = []
    
    for tid, f_list in trials_dict.items():
        n = len(f_list)
        for idx_in_trial, r in enumerate(f_list):
            active_pilot.append(r)
            if idx_in_trial < 3 or idx_in_trial >= n - 3:
                r["_partition"] = "transition"
                trans_pilot.append(r)
            else:
                r["_partition"] = "stable_hold"
                stable_pilot.append(r)

    print(f"Loaded Pilot X: {len(pilot_records)} total records, {len(active_pilot)} active, {len(stable_pilot)} stable, {len(trans_pilot)} transition, {len(trials_dict)} trials")

    # 2. Load fixture X
    fixture_x_path = "backend/data/alphabet_v2/static_landmarks/X.json"
    fixture_x_data = load_fixture(fixture_x_path)
    fixture_x_samples = fixture_x_data.get("samples", [])
    valid_fixture_x = []
    for s in fixture_x_samples:
        lm = s.get("landmarks")
        if not lm or len(lm) != 21:
            continue
        feat = classifier.extract_features(lm)
        if feat:
            valid_fixture_x.append({"sample": s, "feat": feat, "lm": lm})
    print(f"Loaded Fixture X: {len(valid_fixture_x)} valid frames (out of {len(fixture_x_samples)})")

    # Phân loại cohort fixture X
    upright_fixture_x = [f for f in valid_fixture_x if f["feat"]["idx_tip_y"] <= -0.20]
    forward_fixture_x = [f for f in valid_fixture_x if f["feat"]["idx_tip_y"] >= 0.35]
    other_fixture_x = [f for f in valid_fixture_x if f not in upright_fixture_x and f not in forward_fixture_x]
    print(f"Fixture X cohorts: {len(upright_fixture_x)} upright (target f36-f48), {len(forward_fixture_x)} forward (target f108-f136), {len(other_fixture_x)} intermediate")

    # 3. Load all other fixtures
    fixture_files = sorted(glob.glob("backend/data/alphabet_v2/static_landmarks/*.json"))
    other_fixtures = {}
    for fpath in fixture_files:
        fname = os.path.basename(fpath).replace(".json", "")
        if fname == "X":
            continue
        fdata = load_fixture(fpath)
        samples = fdata.get("samples", [])
        valid_samples = []
        for s in samples:
            lm = s.get("landmarks")
            if not lm or len(lm) != 21:
                continue
            feat = classifier.extract_features(lm)
            if feat:
                valid_samples.append({"sample": s, "feat": feat, "lm": lm})
        other_fixtures[fname] = valid_samples
    print(f"Loaded {len(other_fixtures)} other fixture letters")

    # 4. Định nghĩa các Candidate
    candidates = {
        "Baseline (Current Prod)": {
            "d_iw_max": 1.10, "d_tw_max": 0.95, "d_pw_max": 0.60, "d_ti_max": 0.25,
            "desc": "Ngưỡng production hiện tại (overfit D0517B)"
        },
        "Candidate 1 (Conservative)": {
            "d_iw_max": 1.50, "d_tw_max": 1.25, "d_pw_max": 0.65, "d_ti_max": 0.25,
            "desc": "Nới an toàn p95 webcam, giữ d_TI=0.25 chuẩn"
        },
        "Candidate 2 (High-Coverage Balanced)": {
            "d_iw_max": 1.55, "d_tw_max": 1.30, "d_pw_max": 0.65, "d_ti_max": 0.25,
            "desc": "Bao phủ 100% d_IW/d_TW/d_PW của webcam, giữ d_TI=0.25"
        },
        "Candidate 3 (Full-Coverage incl. rep01)": {
            "d_iw_max": 1.55, "d_tw_max": 1.30, "d_pw_max": 0.70, "d_ti_max": 0.30,
            "desc": "Bao phủ cả trial ban đầu rep01 với d_TI=0.30"
        },
        "Candidate 4 (Over-Relaxed Test)": {
            "d_iw_max": 1.60, "d_tw_max": 1.35, "d_pw_max": 0.75, "d_ti_max": 0.30,
            "desc": "Thử nghiệm biên độ rộng để đo áp lực False Positive"
        }
    }

    # 5. Đánh giá từng Candidate trên các tập dữ liệu
    results = {}
    for cname, cparams in candidates.items():
        iw_m = cparams["d_iw_max"]
        tw_m = cparams["d_tw_max"]
        pw_m = cparams["d_pw_max"]
        ti_m = cparams["d_ti_max"]

        # 5.1 Đánh giá trên Pilot Webcam X
        pilot_up_matches = 0
        pilot_fwd_matches = 0
        pilot_total_matches = 0
        stable_matches = 0
        trans_matches = 0
        trial_stats = defaultdict(lambda: {"total": 0, "matched": 0, "stable_total": 0, "stable_matched": 0})

        for r in active_pilot:
            feat = r.get("geometric", {}).get("features", {})
            trial_id = r.get("trial_id", "unknown")
            is_stable = (r.get("_partition") == "stable_hold")

            is_up = eval_x_upright_custom(feat, ti_m, iw_m, tw_m, pw_m)
            is_fwd = eval_x_forward(feat)
            is_m = is_up or is_fwd

            trial_stats[trial_id]["total"] += 1
            if is_stable:
                trial_stats[trial_id]["stable_total"] += 1

            if is_m:
                pilot_total_matches += 1
                trial_stats[trial_id]["matched"] += 1
                if is_up:
                    pilot_up_matches += 1
                if is_fwd:
                    pilot_fwd_matches += 1
                if is_stable:
                    stable_matches += 1
                    trial_stats[trial_id]["stable_matched"] += 1
                else:
                    trans_matches += 1

        trials_pass_50 = sum(1 for t in trial_stats.values() if t["stable_total"] > 0 and (t["stable_matched"] / t["stable_total"]) >= 0.50)
        trials_pass_70 = sum(1 for t in trial_stats.values() if t["stable_total"] > 0 and (t["stable_matched"] / t["stable_total"]) >= 0.70)
        trials_pass_90 = sum(1 for t in trial_stats.values() if t["stable_total"] > 0 and (t["stable_matched"] / t["stable_total"]) >= 0.90)

        # 5.2 Đánh giá trên Fixture X
        fixture_up_matches = sum(1 for f in upright_fixture_x if eval_x_upright_custom(f["feat"], ti_m, iw_m, tw_m, pw_m))
        fixture_fwd_matches = sum(1 for f in forward_fixture_x if eval_x_forward(f["feat"]))
        fixture_total_matches = sum(1 for f in valid_fixture_x if eval_x_match_custom(f["feat"], ti_m, iw_m, tw_m, pw_m))

        # 5.3 Đánh giá False Positives trên tất cả các fixture chữ khác
        fp_by_letter = {}
        total_fps = 0
        evict_fps = defaultdict(int)
        conflict_fps = defaultdict(int)

        for ltr, flist in other_fixtures.items():
            ltr_fp = 0
            for item in flist:
                feat = item["feat"]
                lm = item["lm"]
                if eval_x_match_custom(feat, ti_m, iw_m, tw_m, pw_m):
                    ltr_fp += 1
                    total_fps += 1
                    sim_res = simulate_pipeline(classifier, lm, feat, ti_m, iw_m, tw_m, pw_m)
                    if sim_res["status"] == "MATCH" and sim_res["letter"] == "X":
                        evict_fps[ltr] += 1
                    elif sim_res["status"] == "CONFLICT":
                        conflict_fps[ltr] += 1
            if ltr_fp > 0:
                fp_by_letter[ltr] = ltr_fp

        # 5.4 Kiểm tra riêng biệt P fixture và P webcam holdout
        p_fixture_samples = other_fixtures.get("P", [])
        p_fixture_x_matches = sum(1 for f in p_fixture_samples if eval_x_match_custom(f["feat"], ti_m, iw_m, tw_m, pw_m))

        p_pilot_path = "backend/data/diagnostics/pilot_pilot_20261108_183703.jsonl"
        p_pilot_x_matches = 0
        if os.path.exists(p_pilot_path):
            p_records = load_pilot_jsonl(p_pilot_path)
            for r in p_records:
                if r.get("expected_sign") == "P":
                    p_feat = r.get("geometric", {}).get("features", {})
                    if eval_x_match_custom(p_feat, ti_m, iw_m, tw_m, pw_m):
                        p_pilot_x_matches += 1

        results[cname] = {
            "params": cparams,
            "pilot_active_total": len(active_pilot),
            "pilot_stable_total": len(stable_pilot),
            "pilot_trans_total": len(trans_pilot),
            "pilot_total_matched": pilot_total_matches,
            "pilot_stable_matched": stable_matches,
            "pilot_trans_matched": trans_matches,
            "pilot_stable_recall": (stable_matches / len(stable_pilot)) if stable_pilot else 0.0,
            "pilot_trans_recall": (trans_matches / len(trans_pilot)) if trans_pilot else 0.0,
            "pilot_active_recall": (pilot_total_matches / len(active_pilot)) if active_pilot else 0.0,
            "trials_pass_50": trials_pass_50,
            "trials_pass_70": trials_pass_70,
            "trials_pass_90": trials_pass_90,
            "trial_details": dict(trial_stats),
            "fixture_x_total": len(valid_fixture_x),
            "fixture_upright_total": len(upright_fixture_x),
            "fixture_upright_matched": fixture_up_matches,
            "fixture_upright_recall": (fixture_up_matches / len(upright_fixture_x)) if upright_fixture_x else 0.0,
            "fixture_forward_total": len(forward_fixture_x),
            "fixture_forward_matched": fixture_fwd_matches,
            "fixture_forward_recall": (fixture_fwd_matches / len(forward_fixture_x)) if forward_fixture_x else 0.0,
            "fixture_total_matched": fixture_total_matches,
            "fixture_total_recall": (fixture_total_matches / len(valid_fixture_x)) if valid_fixture_x else 0.0,
            "fp_by_letter": fp_by_letter,
            "total_fps": total_fps,
            "evict_fps": dict(evict_fps),
            "conflict_fps": dict(conflict_fps),
            "p_fixture_x_matches": p_fixture_x_matches,
            "p_pilot_x_matches": p_pilot_x_matches
        }

    # In kết quả dạng bảng
    print("\n" + "="*125)
    print("BẢNG SO SÁNH ĐỊNH LƯỢNG ĐA TIÊU CHÍ CÁC PHƯƠNG ÁN CANDIDATE CHỮ X")
    print("="*125)
    headers = [
        "Candidate", "d_IW/TW/PW/TI", "Stable Rec (228)", "Trans Rec (60)", 
        "Trials >=70%", "Fix Up (12)", "Fix Fwd (11)", "Fix Tot (48)", "FP Letters", "P Overlap"
    ]
    print(f"{headers[0]:<32} | {headers[1]:<17} | {headers[2]:<16} | {headers[3]:<15} | {headers[4]:<12} | {headers[5]:<11} | {headers[6]:<11} | {headers[7]:<11} | {headers[8]:<10} | {headers[9]:<10}")
    print("-" * 155)
    for cname, res in results.items():
        p = res["params"]
        p_str = f"{p['d_iw_max']}/{p['d_tw_max']}/{p['d_pw_max']}/{p['d_ti_max']}"
        st_rec = f"{res['pilot_stable_matched']}/{res['pilot_stable_total']} ({res['pilot_stable_recall']*100:.1f}%)"
        tr_rec = f"{res['pilot_trans_matched']}/{res['pilot_trans_total']} ({res['pilot_trans_recall']*100:.1f}%)"
        tr_70 = f"{res['trials_pass_70']}/10 trials"
        fix_up = f"{res['fixture_upright_matched']}/{res['fixture_upright_total']} ({res['fixture_upright_recall']*100:.1f}%)"
        fix_fwd = f"{res['fixture_forward_matched']}/{res['fixture_forward_total']} ({res['fixture_forward_recall']*100:.1f}%)"
        fix_tot = f"{res['fixture_total_matched']}/{res['fixture_x_total']} ({res['fixture_total_recall']*100:.1f}%)"
        fp_str = f"{res['total_fps']} ({','.join(res['fp_by_letter'].keys()) if res['fp_by_letter'] else 'None'})"
        p_str_ov = f"Fix:{res['p_fixture_x_matches']}, Pil:{res['p_pilot_x_matches']}"
        print(f"{cname:<32} | {p_str:<17} | {st_rec:<16} | {tr_rec:<15} | {tr_70:<12} | {fix_up:<11} | {fix_fwd:<11} | {fix_tot:<11} | {fp_str:<10} | {p_str_ov:<10}")

    print("\n" + "="*90)
    print("CHI TIẾT TỪNG TRIAL (10 TRIALS) TRÊN WEBCAM PILOT X")
    print("="*90)
    for cname, res in results.items():
        print(f"\n--- {cname} ---")
        t_stats = res["trial_details"]
        for tid in sorted(t_stats.keys()):
            s = t_stats[tid]
            pct = (s["stable_matched"] / s["stable_total"] * 100) if s["stable_total"] > 0 else 0.0
            print(f"  {tid}: Stable {s['stable_matched']:2d}/{s['stable_total']:2d} ({pct:5.1f}%) | Total {s['matched']:2d}/{s['total']:2d}")

    # 6. Phân tích quét đơn biến (Single-variable sensitivity sweep)
    print("\n" + "="*90)
    print("PHÂN TÍCH ĐỘ NHẠY ĐƠN BIẾN (SINGLE-VARIABLE SENSITIVITY SWEEP)")
    print("="*90)
    print("\n1. Quét d_IW_MAX (với d_TW=1.30, d_PW=0.65, d_TI=0.25):")
    for iw_val in [1.10, 1.30, 1.40, 1.45, 1.50, 1.55, 1.60]:
        st_m = sum(1 for r in stable_pilot if eval_x_upright_custom(r.get("geometric", {}).get("features", {}), 0.25, iw_val, 1.30, 0.65))
        fps = sum(eval_x_upright_custom(it["feat"], 0.25, iw_val, 1.30, 0.65) for fl in other_fixtures.values() for it in fl)
        print(f"  d_IW_MAX <= {iw_val:.2f}: Stable Match = {st_m:3d}/{len(stable_pilot)} ({st_m/len(stable_pilot)*100:5.1f}%) | Total FPs = {fps}")

    print("\n2. Quét d_TW_MAX (với d_IW=1.55, d_PW=0.65, d_TI=0.25):")
    for tw_val in [0.95, 1.15, 1.20, 1.25, 1.30, 1.35]:
        st_m = sum(1 for r in stable_pilot if eval_x_upright_custom(r.get("geometric", {}).get("features", {}), 0.25, 1.55, tw_val, 0.65))
        fps = sum(eval_x_upright_custom(it["feat"], 0.25, 1.55, tw_val, 0.65) for fl in other_fixtures.values() for it in fl)
        print(f"  d_TW_MAX <= {tw_val:.2f}: Stable Match = {st_m:3d}/{len(stable_pilot)} ({st_m/len(stable_pilot)*100:5.1f}%) | Total FPs = {fps}")

    print("\n3. Quét d_PW_MAX (với d_IW=1.55, d_TW=1.30, d_TI=0.25):")
    for pw_val in [0.60, 0.62, 0.65, 0.70, 0.75]:
        st_m = sum(1 for r in stable_pilot if eval_x_upright_custom(r.get("geometric", {}).get("features", {}), 0.25, 1.55, 1.30, pw_val))
        fps = sum(eval_x_upright_custom(it["feat"], 0.25, 1.55, 1.30, pw_val) for fl in other_fixtures.values() for it in fl)
        print(f"  d_PW_MAX <= {pw_val:.2f}: Stable Match = {st_m:3d}/{len(stable_pilot)} ({st_m/len(stable_pilot)*100:5.1f}%) | Total FPs = {fps}")

    print("\n4. Quét d_TI_MAX (với d_IW=1.55, d_TW=1.30, d_PW=0.65):")
    for ti_val in [0.20, 0.25, 0.28, 0.30, 0.35]:
        st_m = sum(1 for r in stable_pilot if eval_x_upright_custom(r.get("geometric", {}).get("features", {}), ti_val, 1.55, 1.30, 0.65))
        rep01_m = sum(1 for r in stable_pilot if r.get("trial_id") == "rep01" and eval_x_upright_custom(r.get("geometric", {}).get("features", {}), ti_val, 1.55, 1.30, 0.65))
        rep01_tot = sum(1 for r in stable_pilot if r.get("trial_id") == "rep01")
        fps = sum(eval_x_upright_custom(it["feat"], ti_val, 1.55, 1.30, 0.65) for fl in other_fixtures.values() for it in fl)
        print(f"  d_TI_MAX <= {ti_val:.2f}: Stable Match = {st_m:3d}/{len(stable_pilot)} ({st_m/len(stable_pilot)*100:5.1f}%) | rep01 = {rep01_m:2d}/{rep01_tot} | Total FPs = {fps}")

if __name__ == "__main__":
    main()
