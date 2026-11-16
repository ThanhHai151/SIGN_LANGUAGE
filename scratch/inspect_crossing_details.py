import numpy as np
import json
import sys
sys.path.insert(0, '.')
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

data = np.load('backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz', allow_pickle=True)
meta = json.loads(str(data['metadata_json']))
lms = data['landmarks']

r_indices = [i for i, m in enumerate(meta) if m['label'] == 'R' and 22 <= m['frame_index'] <= 36]

print("FRAME-BY-FRAME LANDMARK 8 (INDEX TIP) VS 12 (MIDDLE TIP) FOR R:")
for idx in r_indices:
    m = meta[idx]
    lm = lms[idx]
    f = m['frame_index']
    p5 = lm[5]   # Index MCP
    p8 = lm[8]   # Index Tip
    p9 = lm[9]   # Middle MCP
    p12 = lm[12] # Middle Tip
    
    # Vector MCP5 -> MCP9 (base vector from index to middle)
    base_dx = p9[0] - p5[0]
    base_dy = p9[1] - p5[1]
    
    # Vector Tip8 -> Tip12 (tip vector from index to middle)
    tip_dx = p12[0] - p8[0]
    tip_dy = p12[1] - p8[1]
    
    print(f"Frame {f:2d}: Base (MCP9-MCP5) dx={base_dx:+.4f}, Tip (Tip12-Tip8) dx={tip_dx:+.4f}, "
          f"MCP5_x={p5[0]:.4f}, MCP9_x={p9[0]:.4f}, Tip8_x={p8[0]:.4f}, Tip12_x={p12[0]:.4f}, "
          f"p8_z={p8[2]:+.4f}, p12_z={p12[2]:+.4f}")


