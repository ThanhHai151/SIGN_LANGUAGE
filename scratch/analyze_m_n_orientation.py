import json
import numpy as np

with open('backend/data/alphabet_v2/static_landmarks/M.json', 'r', encoding='utf-8') as f:
    m_data = json.load(f)

with open('backend/data/alphabet_v2/static_landmarks/N.json', 'r', encoding='utf-8') as f:
    n_data = json.load(f)

def inspect_samples(name, samples):
    print(f"\n=================== {name} ({len(samples)} samples) ===================")
    stable = [s for s in samples if s.get('phase') == 'stable_pose' and s.get('quality_status') == 'ok']
    print(f"Stable samples: {len(stable)}")
    
    for s in stable[:3]:
        fidx = s['frame_index']
        lms = s['landmarks']
        wrist = lms[0]
        # Y coordinate relative to wrist: lm[1] - wrist[1]
        # Remember: Y increases downwards! So if Y > wrist[Y], the point is BELOW wrist!
        # If TIP.y > MCP.y, finger is pointing DOWNWARDS!
        print(f"\nSample frame {fidx}:")
        for finger_name, tip, pip, mcp in [
            ("Thumb", 4, 3, 2),
            ("Index", 8, 6, 5),
            ("Middle", 12, 10, 9),
            ("Ring", 16, 14, 13),
            ("Pinky", 20, 18, 17)
        ]:
            tip_y = lms[tip][1] - wrist[1]
            pip_y = lms[pip][1] - wrist[1]
            mcp_y = lms[mcp][1] - wrist[1]
            # dy = tip_y - mcp_y (>0 means TIP is below MCP)
            print(f"  {finger_name:7s}: TIP_y-wrist={tip_y:+.3f}, MCP_y-wrist={mcp_y:+.3f}, (TIP - MCP)_y={tip_y - mcp_y:+.3f}")

inspect_samples("LETTER M (D0504)", m_data['samples'])
inspect_samples("LETTER N (D0505)", n_data['samples'])

