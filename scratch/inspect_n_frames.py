import json

with open('backend/data/alphabet_v2/static_landmarks/N.json', 'r', encoding='utf-8') as f:
    n_data = json.load(f)

for s in n_data['samples']:
    fidx = s['frame_index']
    phase = s.get('phase')
    status = s.get('quality_status')
    lm = s.get('landmarks')
    if not lm:
        continue
    def dist(i, j):
        return ((lm[i][0]-lm[j][0])**2 + (lm[i][1]-lm[j][1])**2 + (lm[i][2]-lm[j][2])**2)**0.5
    
    d_mw = dist(12, 0)
    d_rw = dist(16, 0)
    d_pw = dist(20, 0)
    ratio = d_rw / d_mw if d_mw > 0 else 0
    print(f"N Frame {fidx:2d} ({phase:11s}, {status}): d_MW={d_mw:.4f}, d_RW={d_rw:.4f}, d_RW/d_MW={ratio:.4f}")

