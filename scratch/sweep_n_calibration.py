import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import json
import numpy as np
from backend.models.alphabet_static_head import load_static_alphabet_head, STATIC_CLASSES_22
from backend.vision.feature_extractor import normalize_hand_landmarks

head, _ = load_static_alphabet_head()

# Load canonical fixtures
fixtures = {}
for letter in ['N', 'M', 'A', 'S', 'E', 'T']:
    path = f'backend/data/alphabet_v2/static_landmarks/{letter}.json'
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            fixtures[letter] = json.load(f).get('samples', [])

# Load legacy 551 dataset
npz_path = 'backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz'
legacy_data = np.load(npz_path)
X_legacy = legacy_data['landmarks']
y_legacy = legacy_data['y']
classes = legacy_data['classes']

# Load webcam N log
webcam_n_lines = [json.loads(l) for l in open('backend/data/diagnostics/pilot_eval_29_pilot_20261112_040819_b6dc69_20261112_041644_frames.jsonl', encoding='utf-8')]
webcam_confs = [float(f['head_confidence']) for f in webcam_n_lines if f.get('head_confidence') is not None]

print("=== N CALIBRATION SWEEP SIMULATION ===")

# Test various candidates
candidates = [
    {"name": "Baseline (0.70)", "thresh": 0.70, "min_margin": 0.0},
    {"name": "Cand 1 (0.60)", "thresh": 0.60, "min_margin": 0.0},
    {"name": "Cand 2 (0.50)", "thresh": 0.50, "min_margin": 0.0},
    {"name": "Cand 3 (0.50 + margin 0.05)", "thresh": 0.50, "min_margin": 0.05},
    {"name": "Cand 4 (0.45)", "thresh": 0.45, "min_margin": 0.0},
    {"name": "Cand 5 (0.45 + margin 0.05)", "thresh": 0.45, "min_margin": 0.05},
    {"name": "Cand 6 (0.40)", "thresh": 0.40, "min_margin": 0.0},
    {"name": "Cand 7 (0.40 + margin 0.05)", "thresh": 0.40, "min_margin": 0.05},
    {"name": "Cand 8 (0.40 + margin 0.10)", "thresh": 0.40, "min_margin": 0.10},
]

def eval_head_with_calib(feat_72, target_letter, thresh, min_margin):
    res = head.predict_feature(feat_72[:72], confidence_threshold=0.0)
    top_k = res.get('top_k', [])
    if not top_k:
        return 'UNKNOWN', 0.0
    top1 = top_k[0]
    top1_letter = top1['letter']
    top1_conf = top1['confidence']
    top2_conf = top_k[1]['confidence'] if len(top_k) > 1 else 0.0
    margin = top1_conf - top2_conf
    
    # Specific calibration for N
    if top1_letter == 'N':
        if top1_conf >= thresh and margin >= min_margin:
            return 'N', top1_conf
        return 'UNKNOWN', 0.0
    else:
        # Standard threshold for other letters (0.70 default)
        if top1_conf >= 0.70:
            return top1_letter, top1_conf
        return 'UNKNOWN', 0.0

for cand in candidates:
    thresh = cand['thresh']
    min_margin = cand['min_margin']
    cname = cand['name']
    
    print(f"\n-------------------------------------------------------------")
    print(f"CANDIDATE: {cname} (thresh={thresh}, margin={min_margin})")
    print(f"-------------------------------------------------------------")
    
    # 1. Canonical N (D0505)
    n_correct = 0
    n_m_confusion = 0
    n_samples = fixtures.get('N', [])
    valid_n = 0
    for s in n_samples:
        lms = s.get('landmarks')
        if not lms:
            continue
        valid_n += 1
        feat_72, _ = normalize_hand_landmarks(lms)
        pred, conf = eval_head_with_calib(feat_72, 'N', thresh, min_margin)
        if pred == 'N':
            n_correct += 1
        res_raw = head.predict_feature(feat_72[:72], confidence_threshold=0.0)
        if res_raw['top_k'][0]['letter'] == 'M':
            n_m_confusion += 1
            
    n_recall = (n_correct / valid_n * 100) if valid_n else 0
    print(f"1. Canonical N Recall: {n_correct}/{valid_n} ({n_recall:.1f}%) | N->M raw: {n_m_confusion}")

    # 2. Canonical M (D0504) - Must NOT predict N
    m_fp_as_n = 0
    m_samples = fixtures.get('M', [])
    valid_m = 0
    for s in m_samples:
        lms = s.get('landmarks')
        if not lms:
            continue
        valid_m += 1
        feat_72, _ = normalize_hand_landmarks(lms)
        pred, conf = eval_head_with_calib(feat_72, 'N', thresh, min_margin)
        if pred == 'N':
            m_fp_as_n += 1
    print(f"2. Canonical M -> N False Positives: {m_fp_as_n}/{valid_m} ({m_fp_as_n/valid_m*100:.1f}%)")

    # 3. Canonical A, S, E, T -> N False Positives
    other_fps = {}
    for letter in ['A', 'S', 'E', 'T']:
        samples = fixtures.get(letter, [])
        fp = 0
        v_cnt = 0
        for s in samples:
            lms = s.get('landmarks')
            if not lms:
                continue
            v_cnt += 1
            feat_72, _ = normalize_hand_landmarks(lms)
            pred, conf = eval_head_with_calib(feat_72, 'N', thresh, min_margin)
            if pred == 'N':
                fp += 1
        other_fps[letter] = f"{fp}/{v_cnt}"
    print(f"3. Canonical A, S, E, T -> N False Positives: {other_fps}")

    # 4. Legacy 551 Dataset
    legacy_n_correct = 0
    legacy_n_total = 0
    legacy_fp_as_n = {}
    for i in range(len(X_legacy)):
        lms = X_legacy[i].tolist()
        expected = str(classes[y_legacy[i]])
        feat_72, _ = normalize_hand_landmarks(lms)
        pred, conf = eval_head_with_calib(feat_72, 'N', thresh, min_margin)
        if expected == 'N':
            legacy_n_total += 1
            if pred == 'N':
                legacy_n_correct += 1
        else:
            if pred == 'N':
                legacy_fp_as_n[expected] = legacy_fp_as_n.get(expected, 0) + 1
                
    leg_n_rec = legacy_n_correct / legacy_n_total * 100 if legacy_n_total else 0
    print(f"4. Legacy 551 N Recall: {legacy_n_correct}/{legacy_n_total} ({leg_n_rec:.1f}%) | Non-N -> N False Positives: {legacy_fp_as_n if legacy_fp_as_n else 'ZERO'}")

    # 5. Webcam N Log Recall (estimated via head_confidence distribution)
    webcam_passed = sum(1 for c in webcam_confs if c >= thresh)
    webcam_pct = webcam_passed / len(webcam_confs) * 100 if webcam_confs else 0
    print(f"5. Webcam N Log Frames with conf >= {thresh}: {webcam_passed}/{len(webcam_confs)} ({webcam_pct:.1f}%)")

