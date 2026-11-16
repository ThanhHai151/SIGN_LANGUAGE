import sys
sys.path.insert(0, '.')
import json
import numpy as np
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

clf = VSLAlphabetClassifier()

# Define the exact candidate conditions from TASK 1.3
def evaluate_p_candidate_conditions(feat):
    if not feat or not isinstance(feat, dict):
        return None
    
    d_iw = feat.get('d_IW', 0.0)
    idx_straight = feat.get('idx_straight', 0.0)
    idx_tip_y = feat.get('idx_tip_y', 0.0)
    idx_dy = feat.get('idx_dy', 0.0)
    idx_horiz_ratio = feat.get('idx_horiz_ratio', 1.0)
    
    d_mw = feat.get('d_MW', 0.0)
    mid_straight = feat.get('mid_straight', 0.0)
    mid_tip_y = feat.get('mid_tip_y', 0.0)
    
    d_tm = feat.get('d_TM', 99.0)
    d_ti = feat.get('d_TI', 99.0)
    d_tw = feat.get('d_TW', 0.0)
    
    ang_im_dir = feat.get('ang_IM_dir', 0.0)
    
    d_rw = feat.get('d_RW', 99.0)
    d_pw = feat.get('d_PW', 99.0)
    
    conds = {
        'c1_d_iw_ge_1_70': bool(d_iw >= 1.70),
        'c2_idx_straight_ge_0_95': bool(idx_straight >= 0.95),
        'c3_idx_tip_y_ge_0_60': bool(idx_tip_y >= 0.60),
        'c4_idx_dy_ge_0_60': bool(idx_dy >= 0.60),
        'c5_idx_horiz_le_0_65': bool(idx_horiz_ratio <= 0.65),
        'c6_d_mw_ge_1_35': bool(d_mw >= 1.35),
        'c7_mid_straight_ge_0_80': bool(mid_straight >= 0.80),
        'c8_mid_tip_y_ge_0_60': bool(mid_tip_y >= 0.60),
        'c9_d_tm_le_0_90': bool(d_tm <= 0.90),
        'c10_d_ti_le_1_00': bool(d_ti <= 1.00),
        'c11_d_tw_range_1_30_2_05': bool(1.30 <= d_tw <= 2.05),
        'c12_ang_im_range_15_50': bool(15.0 <= ang_im_dir <= 50.0),
        'c13_d_rw_le_1_60': bool(d_rw <= 1.60),
        'c14_d_pw_le_1_50': bool(d_pw <= 1.50),
    }
    
    matched = all(conds.values())
    return {
        'matched': matched,
        'conds': conds,
        'failed': [k for k, v in conds.items() if not v]
    }

# Load datasets
datasets = {}

# 1. Webcam P (90 active frames)
log_path = 'backend/data/diagnostics/pilot_pilot_20261108_044024.jsonl'
webcam_p = []
with open(log_path, 'r', encoding='utf-8') as f:
    for line in f:
        r = json.loads(line)
        if r.get('expected_sign') == 'P' and 'P_rep' in r.get('trial_id', ''):
            webcam_p.append({
                'id': f"{r.get('trial_id')}_f{r.get('frame_idx')}",
                'phase': 'active_trial',
                'features': r['geometric']['features']
            })
datasets['Webcam_P'] = webcam_p

# 2. Fixture datasets from static_landmarks
for letter in ['P', 'K', 'Q', 'D']:
    path = f'backend/data/alphabet_v2/static_landmarks/{letter}.json'
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    frames = []
    for s in data.get('samples', []):
        f_idx = s.get('frame_index') or s.get('frame_idx')
        phase = s.get('phase')
        lms = s.get('landmarks')
        if lms:
            feat = clf.extract_features(lms)
            if feat:
                frames.append({
                    'id': f"{letter}_f{f_idx}",
                    'frame_idx': f_idx,
                    'phase': phase,
                    'features': feat
                })
    datasets[f'Fixture_{letter}_all'] = frames
    datasets[f'Fixture_{letter}_stable'] = [f for f in frames if f['phase'] == 'stable_pose']

print('Loaded datasets:')
for k, v in datasets.items():
    print(f'  {k:22s}: {len(v)} frames')

# Run condition-by-condition evaluation
for dname, dlist in datasets.items():
    total = len(dlist)
    if total == 0: continue
    
    cond_counts = {c: 0 for c in [
        'c1_d_iw_ge_1_70', 'c2_idx_straight_ge_0_95', 'c3_idx_tip_y_ge_0_60',
        'c4_idx_dy_ge_0_60', 'c5_idx_horiz_le_0_65', 'c6_d_mw_ge_1_35',
        'c7_mid_straight_ge_0_80', 'c8_mid_tip_y_ge_0_60', 'c9_d_tm_le_0_90',
        'c10_d_ti_le_1_00', 'c11_d_tw_range_1_30_2_05', 'c12_ang_im_range_15_50',
        'c13_d_rw_le_1_60', 'c14_d_pw_le_1_50'
    ]}
    
    matched_list = []
    
    for item in dlist:
        res = evaluate_p_candidate_conditions(item['features'])
        for ck, cv in res['conds'].items():
            if cv: cond_counts[ck] += 1
        if res['matched']:
            matched_list.append(item)
            
    print(f'\n========================================================================')
    print(f'DATASET: {dname} (Total: {total} frames)')
    print(f'Candidate P-down Branch Matched: {len(matched_list)} / {total} ({len(matched_list)/total*100:.1f}%)')
    print('Condition breakdown:')
    for ck, cnt in cond_counts.items():
        print(f'  {ck:26s}: PASS = {cnt:2d}/{total} ({cnt/total*100:5.1f}%) | FAIL = {total-cnt:2d}/{total}')
        
    if matched_list:
        print(f'  MATCHED FRAMES DETAILS:')
        for m in matched_list:
            feat = m['features']
            print(f"    - ID: {m['id']} (phase={m.get('phase')}): tip_y={feat.get('idx_tip_y'):.2f}, dy={feat.get('idx_dy'):.2f}, mid_str={feat.get('mid_straight'):.2f}, ang_IM={feat.get('ang_IM_dir'):.1f}, TM={feat.get('d_TM'):.2f}, TI={feat.get('d_TI'):.2f}, TW={feat.get('d_TW'):.2f}")

