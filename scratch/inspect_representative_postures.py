import json
import numpy as np

for letter, fn in [('T', 'backend/data/alphabet_v2/static_landmarks/T.json'),
                   ('U', 'backend/data/alphabet_v2/static_landmarks/U.json'),
                   ('S', 'backend/data/alphabet_v2/static_landmarks/S.json')]:
    with open(fn, 'r', encoding='utf-8') as f:
        d = json.load(f)
    samples = [s for s in d['samples'] if s.get('phase') == 'stable_pose' and s.get('landmarks')]
    s = samples[len(samples)//2] # representative frame
    lm = np.array(s['landmarks'])
    wrist = lm[0]
    sh = lm - wrist
    ps = np.linalg.norm(sh[9])
    sh_norm = sh / ps
    print(f"\n==================== {letter} Representative Frame {s['frame_index']} (video {d.get('video_id')}) ====================")
    print(f"palm_scale: {ps:.4f}")
    names = [
        "Wrist (0)",
        "Thumb CMC (1)", "Thumb MCP (2)", "Thumb IP (3)", "Thumb TIP (4)",
        "Index MCP (5)", "Index PIP (6)", "Index DIP (7)", "Index TIP (8)",
        "Middle MCP (9)", "Middle PIP (10)", "Middle DIP (11)", "Middle TIP (12)",
        "Ring MCP (13)", "Ring PIP (14)", "Ring DIP (15)", "Ring TIP (16)",
        "Pinky MCP (17)", "Pinky PIP (18)", "Pinky DIP (19)", "Pinky TIP (20)"
    ]
    for i, name in enumerate(names):
        print(f"  {name:<16}: x={sh_norm[i,0]:+.3f}, y={sh_norm[i,1]:+.3f}, z={sh_norm[i,2]:+.3f}")

