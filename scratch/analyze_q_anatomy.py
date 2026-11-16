import sys
sys.path.insert(0, '.')
import json
import numpy as np

with open("backend/data/alphabet_v2/static_landmarks/Q.json", 'r', encoding='utf-8') as f:
    q_data = json.load(f)

for f_idx in [32, 36, 40, 44, 48, 52, 56]:
    s = next((x for x in q_data['samples'] if x.get('frame_index') == f_idx), None)
    if not s or not s.get('landmarks'):
        continue
    lm = np.array(s['landmarks'])
    wrist = lm[0]
    scale = np.linalg.norm(lm[9] - wrist)
    
    # Vector from wrist to MCP 5 (index base)
    v_w_mcp5 = (lm[5] - wrist) / scale
    # Vector from MCP 5 to Tip 8 (index finger)
    v_mcp5_tip8 = (lm[8] - lm[5]) / scale
    # Vector from wrist to Tip 8
    v_w_tip8 = (lm[8] - wrist) / scale
    # Vector from MCP 1 to Tip 4 (thumb)
    v_mcp1_tip4 = (lm[4] - lm[1]) / scale
    
    # Other fingers: mid, ring, pinky tips to MCP
    d_mid = np.linalg.norm(lm[12] - lm[9]) / scale
    d_ring = np.linalg.norm(lm[16] - lm[13]) / scale
    d_pinky = np.linalg.norm(lm[20] - lm[17]) / scale
    
    d_ti = np.linalg.norm(lm[4] - lm[8]) / scale
    
    print(f"Frame {f_idx} (scale={scale:.4f}):")
    print(f"  Wrist->MCP5: (dx={v_w_mcp5[0]:+.2f}, dy={v_w_mcp5[1]:+.2f}, dz={v_w_mcp5[2]:+.2f})")
    print(f"  Index (MCP5->Tip8): (dx={v_mcp5_tip8[0]:+.2f}, dy={v_mcp5_tip8[1]:+.2f}, dz={v_mcp5_tip8[2]:+.2f}) [len={np.linalg.norm(v_mcp5_tip8):.2f}]")
    print(f"  Thumb (MCP1->Tip4): (dx={v_mcp1_tip4[0]:+.2f}, dy={v_mcp1_tip4[1]:+.2f}, dz={v_mcp1_tip4[2]:+.2f}) [len={np.linalg.norm(v_mcp1_tip4):.2f}]")
    print(f"  d_TI={d_ti:.2f}, Curled tips dist: mid={d_mid:.2f}, ring={d_ring:.2f}, pinky={d_pinky:.2f}")

