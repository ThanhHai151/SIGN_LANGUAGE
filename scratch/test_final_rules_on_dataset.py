import os
import glob
import json
import sys
sys.path.insert(0, '.')
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

clf = VSLAlphabetClassifier()

json_files = glob.glob('backend/data/alphabet_v2/static_landmarks/*.json')
target_letters = ['O', 'S', 'P', 'Q']
results = {let: {"TP": 0, "FP": 0, "FN": 0, "TN": 0} for let in target_letters}
fp_details = {let: [] for let in target_letters}

total_samples = 0
for fpath in json_files:
    fname = os.path.basename(fpath)
    let_name = fname.replace('.json', '')
    true_letter = "H" if let_name == "D0500N" else let_name
    
    with open(fpath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    samples = data.get('samples', [])
    for s in samples:
        landmarks = s.get('landmarks')
        if not landmarks or len(landmarks) < 21:
            continue
        total_samples += 1
        feat = clf.extract_features(landmarks)
        if feat is None:
            continue
        
        t_ext = feat["thumb_ext"]
        i_ext = feat["index_ext"]
        m_ext = feat["middle_ext"]
        r_ext = feat["ring_ext"]
        p_ext = feat["pinky_ext"]
        d_ti = feat["d_TI"]
        d_tm = feat["d_TM"]
        d_tw = feat["d_TW"]
        d_iw = feat["d_IW"]
        d_mw = feat["d_MW"]
        d_rw = feat.get("d_RW", 0.0)
        d_pw = feat.get("d_PW", 0.0)
        thumb_lat = feat["thumb_lat"]
        idx_straight = feat.get("idx_straight", 0.0)
        idx_tip_y = feat.get("idx_tip_y", 0.0)
        idx_horiz_ratio = feat.get("idx_horiz_ratio", 0.0)
        ang_im_dir = feat.get("ang_IM_dir", 0.0)
        mid_straight = feat.get("mid_straight", 1.0)
        m_curl = feat.get("middle_curl", 0.0)
        ang_ti = feat.get("ang_TI", 0.0)
        palm_scale = feat["palm_scale"]
        vy_58 = float((landmarks[8][1] - landmarks[5][1]) / palm_scale)

        is_o = (
            not i_ext and not m_ext and not r_ext and not p_ext
            and d_ti <= 0.22 and d_tm <= 0.22
            and d_iw >= 0.95 and d_mw >= 0.90
            and d_rw >= 0.80 and d_pw >= 0.75
            and idx_straight <= 0.40
        )
        is_s = (
            not i_ext and not m_ext and not r_ext and not p_ext
            and d_tw <= 0.95 and 0.25 <= d_ti <= 0.45 and 0.40 <= d_tm <= 0.65
            and d_iw <= 0.90 and d_mw <= 0.75 and d_rw <= 0.70 and d_pw <= 0.70
            and thumb_lat >= 0.50
        )
        is_p = (
            d_iw >= 1.70 and idx_straight >= 0.95 and idx_tip_y <= -1.60 and idx_horiz_ratio <= 0.20
            and d_mw <= 1.25 and d_ti >= 1.35 and d_tm <= 0.55 and d_tw <= 1.05 and thumb_lat >= 0.65
            and ang_im_dir >= 105.0
        )
        is_q = (
            1.80 <= d_iw <= 2.25 and idx_straight >= 0.90
            and 0.30 <= idx_tip_y <= 1.20 and vy_58 >= 0.45 and idx_horiz_ratio >= 0.50
            and 1.25 <= d_tw <= 1.60 and 0.50 <= d_ti <= 0.85 and ang_ti <= 15.0
            and mid_straight <= 0.50 and m_curl >= 0.40 and ang_im_dir >= 30.0
        )

        preds = {"O": is_o, "S": is_s, "P": is_p, "Q": is_q}
        for let in target_letters:
            pred = preds[let]
            is_true = (true_letter == let and s.get('phase') == 'stable_pose')
            if pred and true_letter == let:
                results[let]["TP"] += 1
            elif pred and true_letter != let:
                results[let]["FP"] += 1
                fp_details[let].append((true_letter, s.get('frame_index') or s.get('frame_idx')))
            elif not pred and is_true:
                results[let]["FN"] += 1
            else:
                results[let]["TN"] += 1

print(f"Evaluated across {total_samples} total frames in all 23 JSON files.")
print("\n--- PERFORMANCE SUMMARY ---")
for let in target_letters:
    tp = results[let]["TP"]
    fp = results[let]["FP"]
    fn = results[let]["FN"]
    tn = results[let]["TN"]
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    print(f"Letter {let}: TP={tp}, FP={fp}, FN={fn}, TN={tn} | Prec={prec*100:.1f}%, Rec={rec*100:.1f}%, F1={f1*100:.1f}%")
    if fp > 0:
        print(f"  FP details for {let}: {fp_details[let]}")
