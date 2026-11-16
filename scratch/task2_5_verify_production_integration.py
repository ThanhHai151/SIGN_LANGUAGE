"""
TASK 2.5: Script kiểm tra xác nhận sau khi tích hợp Candidate 2B vào production.
Import trực tiếp AlphabetClassifier từ backend.vision.alphabet_classifier.
KHÔNG monkey patch, KHÔNG class cloning, KHÔNG dummy_feat.
"""
import os
import sys
import glob
import json
import hashlib
from collections import defaultdict, Counter

sys.path.insert(0, os.path.abspath("."))
# Import TRỰC TIẾP từ production
from backend.vision.alphabet_classifier import AlphabetClassifier

def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=" * 85)
    print("TASK 2.5 — KIỂM TRA XÁC NHẬN PRODUCTION SAU KHI TÍCH HỢP CANDIDATE 2B")
    print("=" * 85)

    classifier = AlphabetClassifier()
    print("Đã khởi tạo instance trực tiếp từ AlphabetClassifier production.")

    # 1. Kiểm tra hằng số production
    from backend.vision.alphabet_classifier import (
        X_D_TI_MAX, X_D_IW_MAX, X_IDX_CURL_MIN, X_IDX_STRAIGHT_MIN,
        X_IDX_STRAIGHT_MAX, X_D_RW_MAX, X_D_PW_MAX, X_D_TW_MIN, X_D_TW_MAX,
        X_IDX_TIP_Y_MAX
    )
    print("\n1. Kiểm tra hằng số production:")
    print(f"  X_D_TI_MAX         = {X_D_TI_MAX} (Chuẩn: 0.25)")
    print(f"  X_D_IW_MAX         = {X_D_IW_MAX} (Chuẩn: 1.55)")
    print(f"  X_IDX_CURL_MIN     = {X_IDX_CURL_MIN} (Chuẩn: 0.15)")
    print(f"  X_IDX_STRAIGHT_MIN = {X_IDX_STRAIGHT_MIN} (Chuẩn: 0.40)")
    print(f"  X_IDX_STRAIGHT_MAX = {X_IDX_STRAIGHT_MAX} (Chuẩn: 0.85)")
    print(f"  X_D_RW_MAX         = {X_D_RW_MAX} (Chuẩn: 0.70)")
    print(f"  X_D_PW_MAX         = {X_D_PW_MAX} (Chuẩn: 0.65)")
    print(f"  X_D_TW_MIN         = {X_D_TW_MIN} (Chuẩn: 0.70)")
    print(f"  X_D_TW_MAX         = {X_D_TW_MAX} (Chuẩn: 1.30)")
    print(f"  X_IDX_TIP_Y_MAX    = {X_IDX_TIP_Y_MAX} (Chuẩn: -0.20)")
    assert X_D_IW_MAX == 1.55
    assert X_D_TW_MAX == 1.30
    assert X_D_PW_MAX == 0.65
    assert X_IDX_STRAIGHT_MIN == 0.40

    # 2. Kiểm tra trên Webcam Pilot X
    pilot_path = "backend/data/diagnostics/pilot_pilot_20261108_184115.jsonl"
    records = []
    with open(pilot_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))

    trials_dict = defaultdict(list)
    for r in records:
        if r.get("expected_sign") == "X" and r.get("trial_id") is not None:
            trials_dict[r.get("trial_id")].append(r)

    active_frames = []
    stable_frames = []
    trans_frames = []
    trial_stats = defaultdict(lambda: {"tot": 0, "m": 0, "stable_tot": 0, "stable_m": 0})

    for tid, f_list in trials_dict.items():
        n = len(f_list)
        for idx, r in enumerate(f_list):
            active_frames.append(r)
            trial_stats[tid]["tot"] += 1
            feat = r["geometric"]["features"]
            # Gọi TRỰC TIẾP evaluate_rules production
            matches = classifier.evaluate_rules(feat)
            is_matched_x = any(m[0] == "X" for m in matches)
            if is_matched_x:
                trial_stats[tid]["m"] += 1

            if idx < 3 or idx >= n - 3:
                r["_part"] = "transition"
                trans_frames.append(r)
            else:
                r["_part"] = "stable_hold"
                stable_frames.append(r)
                trial_stats[tid]["stable_tot"] += 1
                if is_matched_x:
                    trial_stats[tid]["stable_m"] += 1

    pilot_active_m = sum(trial_stats[t]["m"] for t in trial_stats)
    pilot_stable_m = sum(trial_stats[t]["stable_m"] for t in trial_stats)
    pilot_trans_m = pilot_active_m - pilot_stable_m
    trials_pass_70 = sum(1 for t in trial_stats.values() if t["stable_tot"] > 0 and (t["stable_m"] / t["stable_tot"]) >= 0.70)
    rep02_10_tot = sum(t["stable_tot"] for tid, t in trial_stats.items() if tid != "pilot_20261108_184115_X_rep01")
    rep02_10_m = sum(t["stable_m"] for tid, t in trial_stats.items() if tid != "pilot_20261108_184115_X_rep01")

    print("\n2. Kiểm tra trên Webcam Pilot X:")
    print(f"  Stable Hold Recall : {pilot_stable_m}/228 ({pilot_stable_m/228*100:.2f}%) [TASK 2.4.1: 206/228 = 90.35%]")
    print(f"  Transition Recall  : {pilot_trans_m}/60 ({pilot_trans_m/60*100:.2f}%) [TASK 2.4.1: 54/60 = 90.00%]")
    print(f"  Active Recall      : {pilot_active_m}/288 ({pilot_active_m/288*100:.2f}%) [TASK 2.4.1: 260/288 = 90.28%]")
    print(f"  Trials >= 70%      : {trials_pass_70}/10 [TASK 2.4.1: 9/10]")
    print(f"  Nhóm rep02-rep10   : {rep02_10_m}/{rep02_10_tot} ({rep02_10_m/rep02_10_tot*100:.2f}%) [TASK 2.4.1: 200/201 = 99.50%]")

    assert pilot_stable_m == 206, f"Sai khác stable: {pilot_stable_m} != 206"
    assert pilot_trans_m == 54, f"Sai khác trans: {pilot_trans_m} != 54"
    assert pilot_active_m == 260, f"Sai khác active: {pilot_active_m} != 260"
    assert trials_pass_70 == 9
    assert rep02_10_m == 200

    # 3. Kiểm tra trên Fixture X
    fx_path = "backend/data/alphabet_v2/static_landmarks/X.json"
    with open(fx_path, "r", encoding="utf-8") as f:
        fx_data = json.load(f)

    fx_valid = []
    fx_upright = []
    fx_forward = []
    for s in fx_data.get("samples", []):
        lm = s.get("landmarks")
        if not lm or len(lm) != 21: continue
        feat = classifier.extract_features(lm)
        if feat:
            item = {"s": s, "feat": feat, "lm": lm}
            fx_valid.append(item)
            if feat["idx_tip_y"] <= -0.20:
                fx_upright.append(item)
            elif feat["idx_tip_y"] >= 0.35:
                fx_forward.append(item)

    # Chạy trực tiếp classify_detailed trên landmark
    fx_tot_m = sum(1 for item in fx_valid if classifier.classify_detailed(item["lm"])["letter"] == "X")
    fx_up_m = sum(1 for item in fx_upright if classifier.classify_detailed(item["lm"])["letter"] == "X")
    fx_fwd_m = sum(1 for item in fx_forward if classifier.classify_detailed(item["lm"])["letter"] == "X")

    print("\n3. Kiểm tra trên Fixture X:")
    print(f"  Fixture X Total    : {fx_tot_m}/48 ({fx_tot_m/48*100:.2f}%) [TASK 2.4.1: 25/48 = 52.08%]")
    print(f"  Fixture X Upright  : {fx_up_m}/20 ({fx_up_m/20*100:.2f}%) [TASK 2.4.1: 13/20 = 65.00%]")
    print(f"  Fixture X Forward  : {fx_fwd_m}/27 ({fx_fwd_m/27*100:.2f}%) [TASK 2.4.1: 12/27 = 44.44%]")
    assert fx_tot_m == 25
    assert fx_up_m == 13
    assert fx_fwd_m == 12

    # 4. Kiểm tra trên 22 Fixtures đối kháng
    print("\n4. Kiểm tra trên 22 Fixtures đối kháng (486 valid frames):")
    tot_fix_fps = 0
    fix_fps = {}
    fixture_files = sorted(glob.glob("backend/data/alphabet_v2/static_landmarks/*.json"))
    for fpath in fixture_files:
        ltr = os.path.basename(fpath).replace(".json", "")
        if ltr == "X": continue
        with open(fpath, "r", encoding="utf-8") as f:
            d = json.load(f)
        ltr_fps = 0
        for s in d.get("samples", []):
            lm = s.get("landmarks")
            if not lm or len(lm) != 21: continue
            res = classifier.classify_detailed(lm)
            if res["letter"] == "X":
                ltr_fps += 1
                tot_fix_fps += 1
        if ltr_fps > 0:
            fix_fps[ltr] = ltr_fps

    print(f"  False Positives thành X: {tot_fix_fps} ({fix_fps}) [TASK 2.4.1: 0]")
    assert tot_fix_fps == 0, f"Có false positives trên fixture: {fix_fps}"

    # 5. Kiểm tra trên Dataset NPZ 551 mẫu
    print("\n5. Kiểm tra trên Dataset NPZ (551 mẫu):")
    npz_path = "backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz"
    import numpy as np
    data_npz = np.load(npz_path, allow_pickle=True)
    classes = data_npz["classes"]
    y = data_npz["y"]
    landmarks = data_npz["landmarks"]

    tot_npz_fps = 0
    npz_fps = {}
    for idx in range(len(y)):
        true_l = classes[y[idx]]
        if true_l == "X": continue
        lm = landmarks[idx].tolist()
        pred_letter, _ = classifier.classify(lm)
        if pred_letter == "X":
            npz_fps[true_l] = npz_fps.get(true_l, 0) + 1
            tot_npz_fps += 1

    print(f"  False Positives thành X: {tot_npz_fps} ({npz_fps}) [TASK 2.4.1: 0]")
    assert tot_npz_fps == 0, f"Có false positives trên NPZ: {npz_fps}"

    # 6. Kiểm tra hồi quy P-down (Webcam holdout P)
    print("\n6. Kiểm tra hồi quy trên Webcam Holdout P:")
    p_pilot_path = "backend/data/diagnostics/pilot_pilot_20261108_183703.jsonl"
    p_x_fps = 0
    p_total = 0
    p_correct = 0
    if os.path.exists(p_pilot_path):
        with open(p_pilot_path, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("expected_sign") == "P":
                    p_total += 1
                    feat = r.get("geometric", {}).get("features")
                    if feat:
                        matches = classifier.evaluate_rules(feat)
                        if any(m[0] == "X" for m in matches):
                            p_x_fps += 1
                        if any(m[0] == "P" for m in matches):
                            p_correct += 1

    print(f"  Số frame P (active)         : {p_total}")
    print(f"  Số frame P bị nhận nhầm là X: {p_x_fps} [TASK 2.4.1: 0]")
    print(f"  Số frame P nhận đúng P     : {p_correct}/{p_total} ({p_correct/p_total*100:.2f}%)")
    assert p_x_fps == 0, f"Có frame P bị nhận nhầm thành X: {p_x_fps}"

    print("\n" + "="*85)
    print("KẾT LUẬN: TOÀN BỘ 6 TIÊU CHÍ TÁI LẬP 100% HOÀN HẢO KHÔNG SAI LỆCH!")
    print("="*85)

if __name__ == "__main__":
    main()

