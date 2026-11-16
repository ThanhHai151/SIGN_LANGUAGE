import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import json
from scratch.test_rule_m import dist_3d

with open('backend/data/alphabet_v2/static_landmarks/N.json', 'r', encoding='utf-8') as f:
    n_data = json.load(f)

with open('backend/data/alphabet_v2/static_landmarks/M.json', 'r', encoding='utf-8') as f:
    m_data = json.load(f)

def print_hand_details(s, name):
    lm = s['landmarks']
    fidx = s['frame_index']
    wrist = lm[0]
    ps = dist_3d(wrist, lm[9])
    d_iw = dist_3d(lm[8], wrist) / ps
    d_mw = dist_3d(lm[12], wrist) / ps
    d_rw = dist_3d(lm[16], wrist) / ps
    d_pw = dist_3d(lm[20], wrist) / ps
    d_mr = dist_3d(lm[12], lm[16]) / ps
    d_im = dist_3d(lm[8], lm[12]) / ps
    d_rp = dist_3d(lm[16], lm[20]) / ps
    
    # Ring finger curl & PIP angle
    # 13 (MCP), 14 (PIP), 15 (DIP), 16 (TIP)
    # calculate angle at PIP (13-14-15) and DIP (14-15-16)
    from backend.vision.feature_extractor import calculate_angle
    ang_r_mcp = calculate_angle(wrist, lm[13], lm[14])
    ang_r_pip = calculate_angle(lm[13], lm[14], lm[15])
    ang_m_pip = calculate_angle(lm[9], lm[10], lm[11])
    
    print(f"{name} f{fidx:02d}: d_MW={d_mw:.2f}, d_RW={d_rw:.2f}, d_MR={d_mr:.2f}, d_RP={d_rp:.2f}, d_PW={d_pw:.2f}, ang_R_PIP={ang_r_pip:.1f}, ang_M_PIP={ang_m_pip:.1f}")

print("=== N SAMPLES ===")
for s in n_data['samples']:
    if s['frame_index'] in [18, 24, 28, 42, 46]:
        print_hand_details(s, "N")

print("\n=== M SAMPLES ===")
for s in m_data['samples']:
    if s['frame_index'] in [46, 48, 50, 52, 54, 56]:
        print_hand_details(s, "M")

