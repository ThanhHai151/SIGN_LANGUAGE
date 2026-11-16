import sys
sys.path.insert(0, '.')
import json
import numpy as np

with open('backend/data/alphabet_v2/static_landmarks/G.json', 'r', encoding='utf-8') as f:
    g_data = json.load(f)

with open('backend/data/alphabet_v2/static_landmarks/Q.json', 'r', encoding='utf-8') as f:
    q_data = json.load(f)

print('=== G dx, dy (MCP5 -> Tip8) in stable frames ===')
for s in g_data['samples']:
    if s.get('phase') == 'stable_pose' and s.get('landmarks'):
        lm = np.array(s['landmarks'])
        scale = np.linalg.norm(lm[9] - lm[0])
        dy = (lm[8][1] - lm[5][1]) / scale
        dx = (lm[8][0] - lm[5][0]) / scale
        print(f"G f{s.get('frame_index')}: dx={dx:.2f}, dy={dy:.2f}")

print('\n=== Q dx, dy (MCP5 -> Tip8) in stable frames ===')
for s in q_data['samples']:
    if s.get('phase') == 'stable_pose' and s.get('landmarks'):
        lm = np.array(s['landmarks'])
        scale = np.linalg.norm(lm[9] - lm[0])
        dy = (lm[8][1] - lm[5][1]) / scale
        dx = (lm[8][0] - lm[5][0]) / scale
        print(f"Q f{s.get('frame_index')}: dx={dx:.2f}, dy={dy:.2f}")

