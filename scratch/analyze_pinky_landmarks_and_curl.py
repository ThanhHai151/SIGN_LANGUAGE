import sys
sys.path.insert(0, '.')
import json
import glob
import math
import numpy as np
from backend.vision.feature_extractor import _shift_and_scale, calculate_distance, calculate_angle, finger_curl_ratio

def analyze_letter(letter, json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    samples = data.get('samples', [])
    print(f"\n{'='*70}")
    print(f"ANALYZING REAL QIPEDC LANDMARKS FOR LETTER {letter} ({len(samples)} samples)")
    print(f"{'='*70}")

    curls = []
    pip_angles = []
    dip_angles = []
    mcp_angles = []
    d_pws = []
    d_tip_mcps = []
    straight_lens = []
    straight_3seg_lens = []

    for s in samples:
        lms = s.get('landmarks')
        if not lms or len(lms) < 21:
            continue
        shifted, palm_scale = _shift_and_scale(lms)
        
        # 17: MCP, 18: PIP, 19: DIP, 20: TIP
        p17 = shifted[17]
        p18 = shifted[18]
        p19 = shifted[19]
        p20 = shifted[20]
        p0  = shifted[0]

        # Current formula (2 segments, skipping DIP 19)
        d_20_17 = calculate_distance(p20, p17)
        d_20_18 = calculate_distance(p20, p18)
        d_18_17 = calculate_distance(p18, p17)
        d_20_19 = calculate_distance(p20, p19)
        d_19_18 = calculate_distance(p19, p18)

        str_len_2seg = d_20_18 + d_18_17
        str_len_3seg = d_20_19 + d_19_18 + d_18_17
        curl_2seg = 1.0 - (d_20_17 / str_len_2seg) if str_len_2seg > 1e-6 else 0.0
        curl_3seg = 1.0 - (d_20_17 / str_len_3seg) if str_len_3seg > 1e-6 else 0.0

        # Angles
        ang_pip = calculate_angle(p17, p18, p19)  # PIP angle
        ang_dip = calculate_angle(p18, p19, p20)  # DIP angle
        ang_mcp = calculate_angle(p0, p17, p18)   # MCP angle
        d_pw = calculate_distance(p20, p0) / palm_scale

        curls.append((curl_2seg, curl_3seg))
        pip_angles.append(ang_pip)
        dip_angles.append(ang_dip)
        mcp_angles.append(ang_mcp)
        d_pws.append(d_pw)
        d_tip_mcps.append(d_20_17 / palm_scale)
        straight_lens.append(str_len_2seg / palm_scale)
        straight_3seg_lens.append(str_len_3seg / palm_scale)

    c2 = [c[0] for c in curls]
    c3 = [c[1] for c in curls]

    print(f"Current curl_2seg: min={min(c2):.4f}, mean={np.mean(c2):.4f}, max={max(c2):.4f}")
    print(f"3-segment curl   : min={min(c3):.4f}, mean={np.mean(c3):.4f}, max={max(c3):.4f}")
    print(f"PIP angle (deg)  : min={min(pip_angles):.1f}, mean={np.mean(pip_angles):.1f}, max={max(pip_angles):.1f}")
    print(f"DIP angle (deg)  : min={min(dip_angles):.1f}, mean={np.mean(dip_angles):.1f}, max={max(dip_angles):.1f}")
    print(f"MCP angle (deg)  : min={min(mcp_angles):.1f}, mean={np.mean(mcp_angles):.1f}, max={max(mcp_angles):.1f}")
    print(f"d_PW / palm_scale: min={min(d_pws):.4f}, mean={np.mean(d_pws):.4f}, max={max(d_pws):.4f}")
    print(f"d(TIP,MCP)/scale : min={min(d_tip_mcps):.4f}, mean={np.mean(d_tip_mcps):.4f}, max={max(d_tip_mcps):.4f}")

for let, p in [('A', 'backend/data/alphabet_v2/static_landmarks/A.json'),
               ('S', 'backend/data/alphabet_v2/static_landmarks/S.json'),
               ('O', 'backend/data/alphabet_v2/static_landmarks/O.json'),
               ('U', 'backend/data/alphabet_v2/static_landmarks/U.json'),
               ('I', 'backend/data/alphabet_v2/static_landmarks/I.json'),
               ('Y', 'backend/data/alphabet_v2/static_landmarks/Y.json'),
               ('B', 'backend/data/alphabet_v2/static_landmarks/B.json')]:
    analyze_letter(let, p)
