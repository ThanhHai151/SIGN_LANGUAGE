import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import json
import numpy as np
import math

from backend.vision.feature_extractor import (
    calculate_distance, calculate_angle, _shift_and_scale,
    finger_curl_ratio, compute_finger_states
)

def analyze_letter(json_path, letter_name):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = data.get('samples', [])
    stable_samples = [s for s in samples if s.get('phase') == 'stable_pose' and s.get('quality_status') == 'ok']
    print(f"\n=======================================================")
    print(f"ANALYSIS FOR LETTER {letter_name}: {len(samples)} total, {len(stable_samples)} stable OK")
    print(f"=======================================================")
    
    records = []
    for s in stable_samples:
        lms = s['landmarks']
        fidx = s['frame_index']
        sh, palm_scale = _shift_and_scale(lms)
        states = compute_finger_states(sh, palm_scale)
        
        # Distances to wrist
        d_tw = calculate_distance(sh[4], sh[0]) / palm_scale
        d_iw = calculate_distance(sh[8], sh[0]) / palm_scale
        d_mw = calculate_distance(sh[12], sh[0]) / palm_scale
        d_rw = calculate_distance(sh[16], sh[0]) / palm_scale
        d_pw = calculate_distance(sh[20], sh[0]) / palm_scale
        
        # Inter-finger tip distances
        d_ti = calculate_distance(sh[4], sh[8]) / palm_scale
        d_tm = calculate_distance(sh[4], sh[12]) / palm_scale
        d_tr = calculate_distance(sh[4], sh[16]) / palm_scale
        d_tp = calculate_distance(sh[4], sh[20]) / palm_scale
        d_im = calculate_distance(sh[8], sh[12]) / palm_scale
        d_mr = calculate_distance(sh[12], sh[16]) / palm_scale
        d_rp = calculate_distance(sh[16], sh[20]) / palm_scale
        
        # Vertical orientations: TIP.y - MCP.y (positive means pointing DOWN)
        # Note: in MediaPipe, Y increases downwards!
        dy_thumb  = (sh[4][1] - sh[2][1]) / palm_scale
        dy_index  = (sh[8][1] - sh[5][1]) / palm_scale
        dy_middle = (sh[12][1] - sh[9][1]) / palm_scale
        dy_ring   = (sh[16][1] - sh[13][1]) / palm_scale
        dy_pinky  = (sh[20][1] - sh[17][1]) / palm_scale
        
        # Joint angles (MCP, PIP)
        ang_idx_mcp = calculate_angle(sh[0], sh[5], sh[6])
        ang_idx_pip = calculate_angle(sh[5], sh[6], sh[7])
        ang_idx_dip = calculate_angle(sh[6], sh[7], sh[8])
        
        ang_mid_mcp = calculate_angle(sh[0], sh[9], sh[10])
        ang_mid_pip = calculate_angle(sh[9], sh[10], sh[11])
        ang_mid_dip = calculate_angle(sh[10], sh[11], sh[12])
        
        ang_rng_mcp = calculate_angle(sh[0], sh[13], sh[14])
        ang_rng_pip = calculate_angle(sh[13], sh[14], sh[15])
        ang_rng_dip = calculate_angle(sh[14], sh[15], sh[16])
        
        ang_pky_mcp = calculate_angle(sh[0], sh[17], sh[18])
        ang_pky_pip = calculate_angle(sh[17], sh[18], sh[19])
        
        records.append({
            'fidx': fidx,
            'd_tw': d_tw, 'd_iw': d_iw, 'd_mw': d_mw, 'd_rw': d_rw, 'd_pw': d_pw,
            'd_ti': d_ti, 'd_tm': d_tm, 'd_tr': d_tr, 'd_tp': d_tp,
            'd_im': d_im, 'd_mr': d_mr, 'd_rp': d_rp,
            'dy_thumb': dy_thumb, 'dy_index': dy_index, 'dy_middle': dy_middle,
            'dy_ring': dy_ring, 'dy_pinky': dy_pinky,
            'i_curl': states['index_curl'], 'm_curl': states['middle_curl'],
            'r_curl': states['ring_curl'], 'p_curl': states['pinky_curl'],
            't_curl': states['thumb_curl'],
            'ang_idx_mcp': ang_idx_mcp, 'ang_idx_pip': ang_idx_pip,
            'ang_mid_mcp': ang_mid_mcp, 'ang_mid_pip': ang_mid_pip,
            'ang_rng_mcp': ang_rng_mcp, 'ang_rng_pip': ang_rng_pip,
            'ang_pky_mcp': ang_pky_mcp, 'ang_pky_pip': ang_pky_pip,
        })
    
    # Statistical summary
    keys = list(records[0].keys())
    keys.remove('fidx')
    stats = {}
    for k in keys:
        vals = [r[k] for r in records]
        stats[k] = {
            'min': min(vals),
            'mean': float(np.mean(vals)),
            'max': max(vals),
            'std': float(np.std(vals))
        }
        print(f"  {k:15s}: min={stats[k]['min']:.4f}, mean={stats[k]['mean']:.4f}, max={stats[k]['max']:.4f}, std={stats[k]['std']:.4f}")
    
    return records, stats

rec_m, stat_m = analyze_letter('backend/data/alphabet_v2/static_landmarks/M.json', 'M')
rec_n, stat_n = analyze_letter('backend/data/alphabet_v2/static_landmarks/N.json', 'N')
