import sys
sys.path.insert(0, ".")
import json
import numpy as np

data = json.load(open('backend/data/alphabet_v2/static_landmarks/I.json', encoding='utf-8'))
samples = [s for s in data['samples'] if s['frame_index'] in (16, 24, 30, 36)]

for s in samples:
    lm = np.array(s['landmarks'])
    f_idx = s['frame_index']
    
    # Palm plane: Wrist (0), Index MCP (5), Pinky MCP (17)
    v1 = lm[5] - lm[0]
    v2 = lm[17] - lm[0]
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)
    
    # Signer facing camera: Camera is at +Z facing -Z or +Z into screen?
    # In MediaPipe, Z is depth relative to wrist. Z is smaller when point is closer to camera.
    print(f"Frame {f_idx}:")
    print(f"  Wrist (0): {lm[0]}")
    print(f"  Index MCP (5): {lm[5]}")
    print(f"  Middle MCP (9): {lm[9]}")
    print(f"  Ring MCP (13): {lm[13]}")
    print(f"  Pinky MCP (17): {lm[17]}")
    print(f"  Normal: {normal}")
    print(f"  Thumb TIP (4): {lm[4]}")
    print(f"  Index TIP (8): {lm[8]}")
    print(f"  Middle TIP (12): {lm[12]}")
    print(f"  Pinky TIP (20): {lm[20]}")

