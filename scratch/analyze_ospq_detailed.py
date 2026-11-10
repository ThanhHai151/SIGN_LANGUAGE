import sys
sys.path.insert(0, '.')
import json
import numpy as np

def analyze_landmarks_anatomy(lm, name):
    lm = np.array(lm)
    # lm: 21 x 3 [x, y, z]
    wrist = lm[0]
    
    # Finger MCPs: [1, 5, 9, 13, 17]
    # Finger TIPs: [4, 8, 12, 16, 20]
    palm_scale = np.linalg.norm(lm[9] - wrist)
    
    print(f"\n=================== ANATOMY FOR {name} (palm_scale={palm_scale:.4f}) ===================")
    fingers = [
        ("Thumb", [1, 2, 3, 4]),
        ("Index", [5, 6, 7, 8]),
        ("Middle", [9, 10, 11, 12]),
        ("Ring", [13, 14, 15, 16]),
        ("Pinky", [17, 18, 19, 20])
    ]
    
    for f_name, idxs in fingers:
        mcp, pip, dip, tip = [lm[i] for i in idxs]
        # Vector from wrist to tip
        v_wrist_tip = (tip - wrist) / palm_scale
        # Vector from mcp to tip
        v_mcp_tip = (tip - mcp) / palm_scale
        # Length
        d_tip_wrist = np.linalg.norm(tip - wrist) / palm_scale
        d_tip_mcp = np.linalg.norm(tip - mcp) / palm_scale
        # Direction: dx, dy, dz (normalized)
        dir_norm = (tip - mcp) / (np.linalg.norm(tip - mcp) + 1e-6)
        
        # Curl ratio: tip to mcp / sum of segments
        seg_len = (np.linalg.norm(pip - mcp) + np.linalg.norm(dip - pip) + np.linalg.norm(tip - dip)) / palm_scale
        straightness = (np.linalg.norm(tip - mcp) / palm_scale) / (seg_len + 1e-6)
        
        print(f"  {f_name:<7}: dist_wrist={d_tip_wrist:.2f}, dist_mcp={d_tip_mcp:.2f}, straight={straightness:.2f}, dir=(dx={dir_norm[0]:+.2f}, dy={dir_norm[1]:+.2f}, dz={dir_norm[2]:+.2f})")

    # Joint angles
    print("  Key joint distances:")
    print(f"    Tip 4 - Tip 8 (Thumb-Index): {np.linalg.norm(lm[4] - lm[8])/palm_scale:.2f}")
    print(f"    Tip 4 - Tip 12 (Thumb-Middle): {np.linalg.norm(lm[4] - lm[12])/palm_scale:.2f}")
    print(f"    Tip 8 - Tip 12 (Index-Middle): {np.linalg.norm(lm[8] - lm[12])/palm_scale:.2f}")
    print(f"    Tip 4 - MCP 9 (Thumb-MiddleMCP): {np.linalg.norm(lm[4] - lm[9])/palm_scale:.2f}")
    print(f"    Tip 4 - MCP 13 (Thumb-RingMCP): {np.linalg.norm(lm[4] - lm[13])/palm_scale:.2f}")

for letter, f_target in [("O", 36), ("S", 36), ("P", 36), ("Q", 40)]:
    with open(f"backend/data/alphabet_v2/static_landmarks/{letter}.json", 'r', encoding='utf-8') as f:
        meta = json.load(f)
    for s in meta['samples']:
        if (s.get('frame_index') or s.get('frame_idx')) == f_target and s.get('landmarks'):
            analyze_landmarks_anatomy(s['landmarks'], f"Letter {letter} (Frame {f_target})")
            break

