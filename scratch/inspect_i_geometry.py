import sys
sys.path.insert(0, ".")
import json
import numpy as np

data = json.load(open('backend/data/alphabet_v2/static_landmarks/I.json', encoding='utf-8'))
samples = [s for s in data['samples'] if s['phase'] == 'stable_pose' and s['landmarks'] is not None]

print(f"Total stable samples: {len(samples)}")
for s in samples:
    lm = np.array(s['landmarks'])
    f_idx = s['frame_index']
    # 0: wrist
    # 1-4: thumb (CMC, MCP, IP, TIP)
    # 5-8: index (MCP, PIP, DIP, TIP)
    # 9-12: middle (MCP, PIP, DIP, TIP)
    # 13-16: ring (MCP, PIP, DIP, TIP)
    # 17-20: pinky (MCP, PIP, DIP, TIP)
    wrist = lm[0]
    p_mcp = lm[9]
    scale = np.linalg.norm(p_mcp - wrist)
    
    # Finger tip relative to wrist (y is down in image coordinates, so -y is up)
    # Let's inspect z coordinates too
    thumb_tip = lm[4]
    index_tip = lm[8]
    mid_tip = lm[12]
    ring_tip = lm[16]
    pinky_tip = lm[20]
    
    # distances normalized
    d_tw = np.linalg.norm(thumb_tip - wrist) / scale
    d_iw = np.linalg.norm(index_tip - wrist) / scale
    d_mw = np.linalg.norm(mid_tip - wrist) / scale
    d_rw = np.linalg.norm(ring_tip - wrist) / scale
    d_pw = np.linalg.norm(pinky_tip - wrist) / scale
    
    # Thumb to fingers
    d_ti = np.linalg.norm(thumb_tip - index_tip) / scale
    d_tm = np.linalg.norm(thumb_tip - mid_tip) / scale
    d_tr = np.linalg.norm(thumb_tip - ring_tip) / scale
    
    # Thumb tip to index/middle MCP or PIP
    index_pip = lm[6]
    mid_pip = lm[10]
    ring_pip = lm[14]
    d_thumb_index_pip = np.linalg.norm(thumb_tip - index_pip) / scale
    d_thumb_mid_pip = np.linalg.norm(thumb_tip - mid_pip) / scale
    
    # Pinky direction vector
    pinky_base = lm[17]
    pinky_pip = lm[18]
    p_vec = pinky_tip - pinky_base
    p_len = np.linalg.norm(p_vec)
    p_horiz_ratio = abs(p_vec[0]) / (p_len + 1e-6)
    p_vert_ratio = (pinky_base[1] - pinky_tip[1]) / (p_len + 1e-6) # > 0 means tip is above base
    
    # Thumb direction vector (from CMC to TIP)
    t_base = lm[1]
    t_vec = thumb_tip - t_base
    t_len = np.linalg.norm(t_vec)
    
    print(f"f{f_idx:02d}: scale={scale:.3f} | d_PW={d_pw:.2f} d_IW={d_iw:.2f} d_MW={d_mw:.2f} d_RW={d_rw:.2f} d_TW={d_tw:.2f} | d_TI={d_ti:.2f} d_TM={d_tm:.2f} d_T_MidPIP={d_thumb_mid_pip:.2f} | Pinky: horiz={p_horiz_ratio:.2f} vert={p_vert_ratio:.2f}")

