import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import json
import numpy as np

with open('backend/data/alphabet_v2/static_landmarks/M.json', 'r', encoding='utf-8') as f:
    m_data = json.load(f)

with open('backend/data/alphabet_v2/static_landmarks/N.json', 'r', encoding='utf-8') as f:
    n_data = json.load(f)

def compute_detailed_metrics(samples, label):
    print(f"\n================ METRICS FOR {label} ================")
    stable = [s for s in samples if s.get('phase') == 'stable_pose' and s.get('quality_status') == 'ok']
    
    ratios_rw_mw = []
    ratios_iw_mw = []
    ratios_pw_mw = []
    ratios_pw_rw = []
    diff_mw_rw = []
    diff_rw_pw = []
    
    # Downward ratios: (tip_y - mcp_y) / length
    idx_down = []
    mid_down = []
    rng_down = []
    pky_down = []
    
    for s in stable:
        lm = s['landmarks']
        # Distance helper
        def dist(i, j):
            return ((lm[i][0]-lm[j][0])**2 + (lm[i][1]-lm[j][1])**2 + (lm[i][2]-lm[j][2])**2)**0.5
        
        d_iw = dist(8, 0)
        d_mw = dist(12, 0)
        d_rw = dist(16, 0)
        d_pw = dist(20, 0)
        d_tw = dist(4, 0)
        
        r_rw_mw = d_rw / d_mw if d_mw > 0 else 0
        r_iw_mw = d_iw / d_mw if d_mw > 0 else 0
        r_pw_mw = d_pw / d_mw if d_mw > 0 else 0
        r_pw_rw = d_pw / d_rw if d_rw > 0 else 0
        
        # Downward verticality: (tip_y - mcp_y) / 2d_dist(mcp, tip)
        def down_ratio(tip, mcp):
            dy = lm[tip][1] - lm[mcp][1]
            l = ((lm[tip][0]-lm[mcp][0])**2 + (lm[tip][1]-lm[mcp][1])**2)**0.5
            return dy / (l + 1e-6)
        
        ratios_rw_mw.append(r_rw_mw)
        ratios_iw_mw.append(r_iw_mw)
        ratios_pw_mw.append(r_pw_mw)
        ratios_pw_rw.append(r_pw_rw)
        diff_mw_rw.append(d_mw - d_rw)
        diff_rw_pw.append(d_rw - d_pw)
        
        idx_down.append(down_ratio(8, 5))
        mid_down.append(down_ratio(12, 9))
        rng_down.append(down_ratio(16, 13))
        pky_down.append(down_ratio(20, 17))
        
    print(f"Total stable: {len(stable)}")
    print(f"  Ratio d_RW / d_MW: min={min(ratios_rw_mw):.4f}, mean={np.mean(ratios_rw_mw):.4f}, max={max(ratios_rw_mw):.4f}, std={np.std(ratios_rw_mw):.4f}")
    print(f"  Ratio d_IW / d_MW: min={min(ratios_iw_mw):.4f}, mean={np.mean(ratios_iw_mw):.4f}, max={max(ratios_iw_mw):.4f}, std={np.std(ratios_iw_mw):.4f}")
    print(f"  Ratio d_PW / d_MW: min={min(ratios_pw_mw):.4f}, mean={np.mean(ratios_pw_mw):.4f}, max={max(ratios_pw_mw):.4f}, std={np.std(ratios_pw_mw):.4f}")
    print(f"  Ratio d_PW / d_RW: min={min(ratios_pw_rw):.4f}, mean={np.mean(ratios_pw_rw):.4f}, max={max(ratios_pw_rw):.4f}, std={np.std(ratios_pw_rw):.4f}")
    print(f"  Index Downward Ratio: min={min(idx_down):.4f}, mean={np.mean(idx_down):.4f}, max={max(idx_down):.4f}")
    print(f"  Middle Downward Ratio: min={min(mid_down):.4f}, mean={np.mean(mid_down):.4f}, max={max(mid_down):.4f}")
    print(f"  Ring Downward Ratio: min={min(rng_down):.4f}, mean={np.mean(rng_down):.4f}, max={max(rng_down):.4f}")
    print(f"  Pinky Downward Ratio: min={min(pky_down):.4f}, mean={np.mean(pky_down):.4f}, max={max(pky_down):.4f}")

compute_detailed_metrics(m_data['samples'], 'LETTER M (D0504)')
compute_detailed_metrics(n_data['samples'], 'LETTER N (D0505)')

