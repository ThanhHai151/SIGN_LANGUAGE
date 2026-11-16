import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import json
from backend.vision.feature_extractor import calculate_angle, calculate_distance, _shift_and_scale

with open('backend/data/alphabet_v2/static_landmarks/N.json', 'r', encoding='utf-8') as f:
    n_data = json.load(f)

with open('backend/data/alphabet_v2/static_landmarks/M.json', 'r', encoding='utf-8') as f:
    m_data = json.load(f)

print("=== CHECKING ALL N SAMPLES ===")
n_matches = 0
for s in n_data['samples']:
    lm = s.get('landmarks')
    if not lm: continue
    fidx = s['frame_index']
    sh, ps = _shift_and_scale(lm)
    d_mr = calculate_distance(sh[12], sh[16]) / ps
    d_mw = calculate_distance(sh[12], sh[0]) / ps
    d_rw = calculate_distance(sh[16], sh[0]) / ps
    ang_r_pip = calculate_angle(sh[13], sh[14], sh[15])
    
    # check condition: d_MR <= 0.38 and ang_r_pip >= 170.0 and abs(d_mw - d_rw) <= 0.35
    is_m_like = (d_mr <= 0.38 and ang_r_pip >= 170.0 and abs(d_mw - d_rw) <= 0.35)
    if is_m_like:
        n_matches += 1
        print(f"  N f{fidx}: d_MR={d_mr:.2f}, ang_R_PIP={ang_r_pip:.1f}, diff_MW_RW={abs(d_mw-d_rw):.2f}")

print(f"Total N frames passing (should be 0): {n_matches} / {len(n_data['samples'])}")

print("\n=== CHECKING ALL M SAMPLES ===")
m_matches = 0
m_stable_matches = 0
m_stable_count = 0
for s in m_data['samples']:
    lm = s.get('landmarks')
    if not lm: continue
    fidx = s['frame_index']
    phase = s.get('phase')
    status = s.get('quality_status')
    sh, ps = _shift_and_scale(lm)
    d_mr = calculate_distance(sh[12], sh[16]) / ps
    d_mw = calculate_distance(sh[12], sh[0]) / ps
    d_rw = calculate_distance(sh[16], sh[0]) / ps
    ang_r_pip = calculate_angle(sh[13], sh[14], sh[15])
    
    is_stable = (phase == 'stable_pose' and status == 'ok')
    if is_stable: m_stable_count += 1
    
    is_m_like = (d_mr <= 0.38 and ang_r_pip >= 170.0 and abs(d_mw - d_rw) <= 0.40)
    if is_m_like:
        m_matches += 1
        if is_stable: m_stable_matches += 1
    else:
        if is_stable:
            print(f"  M stable f{fidx} missed: d_MR={d_mr:.2f}, ang_R_PIP={ang_r_pip:.1f}, diff_MW_RW={abs(d_mw-d_rw):.2f}")

print(f"Total M frames passing: {m_matches} / {len(m_data['samples'])}, stable: {m_stable_matches} / {m_stable_count}")

