import sys
sys.path.insert(0, '.')
import numpy as np
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

clf = VSLAlphabetClassifier()
data = np.load('backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz', allow_pickle=True)
classes = list(data['classes'])
landmarks = data['landmarks']
y = data['y']

print("=== SAMPLES OF G MATCHING REFINED Q ===")
for i in range(len(y)):
    if classes[y[i]] == 'G':
        lm = landmarks[i].tolist()
        feat = clf.extract_features(lm)
        scale = feat['palm_scale']
        dy = (lm[8][1] - lm[5][1]) / scale
        dx = (lm[8][0] - lm[5][0]) / scale
        
        d_iw = feat['d_IW']
        d_mw = feat['d_MW']
        d_tw = feat['d_TW']
        d_ti = feat['d_TI']
        idx_straight = feat['idx_straight']
        mid_straight = feat['mid_straight']
        
        cond_q = (
            d_iw >= 1.80
            and idx_straight >= 0.90
            and dy >= 0.45
            and feat['idx_tip_y'] >= 0.30
            and d_tw >= 1.25
            and 0.20 <= d_ti <= 1.05
            and (d_iw - d_mw) >= 0.55
            and mid_straight <= 0.65
        )
        if cond_q:
            # Let's inspect differences: thumb vector, wrist to MCP vector, MCP width, angles
            # Vector of thumb MCP 1 -> Tip 4:
            thumb_dx = (lm[4][0] - lm[1][0]) / scale
            thumb_dy = (lm[4][1] - lm[1][1]) / scale
            # Vector of wrist -> MCP 5:
            w_mcp5_dx = (lm[5][0] - lm[0][0]) / scale
            w_mcp5_dy = (lm[5][1] - lm[0][1]) / scale
            
            print(f"G sample {i}: dx={dx:.2f}, dy={dy:.2f}, thumb=(dx={thumb_dx:.2f}, dy={thumb_dy:.2f}), "
                  f"w_mcp5=(dx={w_mcp5_dx:.2f}, dy={w_mcp5_dy:.2f}), ang_TI={feat['ang_TI']:.1f}, "
                  f"thumb_lat={feat['thumb_lat']:.2f}, d_TW={d_tw:.2f}")

print("\n=== SAMPLES OF Q (FOR COMPARISON) ===")
for i in range(len(y)):
    if classes[y[i]] == 'Q':
        lm = landmarks[i].tolist()
        feat = clf.extract_features(lm)
        scale = feat['palm_scale']
        dy = (lm[8][1] - lm[5][1]) / scale
        dx = (lm[8][0] - lm[5][0]) / scale
        
        thumb_dx = (lm[4][0] - lm[1][0]) / scale
        thumb_dy = (lm[4][1] - lm[1][1]) / scale
        w_mcp5_dx = (lm[5][0] - lm[0][0]) / scale
        w_mcp5_dy = (lm[5][1] - lm[0][1]) / scale
        
        print(f"Q sample {i}: dx={dx:.2f}, dy={dy:.2f}, thumb=(dx={thumb_dx:.2f}, dy={thumb_dy:.2f}), "
              f"w_mcp5=(dx={w_mcp5_dx:.2f}, dy={w_mcp5_dy:.2f}), ang_TI={feat['ang_TI']:.1f}, "
              f"thumb_lat={feat['thumb_lat']:.2f}, d_TW={feat['d_TW']:.2f}")

