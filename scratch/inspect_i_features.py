import sys
sys.path.insert(0, ".")
import json
from backend.vision.alphabet_classifier import alphabet_classifier

data = json.load(open('backend/data/alphabet_v2/static_landmarks/I.json', encoding='utf-8'))
samples = [s for s in data['samples'] if s['landmarks'] is not None]

print(f"Total non-null samples: {len(samples)}")
for s in samples:
    f_idx = s['frame_index']
    phase = s['phase']
    lm = s['landmarks']
    feat = alphabet_classifier.extract_features(lm)
    if not feat:
        continue
    
    # Check what features look like
    p_curl = feat['pinky_curl']
    p_ext = feat['pinky_ext']
    d_pw = feat['d_PW']
    d_iw = feat['d_IW']
    d_mw = feat['d_MW']
    d_rw = feat['d_RW']
    d_tw = feat['d_TW']
    d_ti = feat['d_TI']
    d_tm = feat['d_TM']
    i_curl = feat['index_curl']
    m_curl = feat['middle_curl']
    r_curl = feat['ring_curl']
    t_ext = feat['thumb_ext']
    i_ext = feat['index_ext']
    m_ext = feat['middle_ext']
    r_ext = feat['ring_ext']
    
    print(f"f{f_idx:02d} ({phase:11s}): d_PW={d_pw:.2f} p_ext={p_ext} p_curl={p_curl:.2f} | d_IW={d_iw:.2f} i_curl={i_curl:.2f} | d_MW={d_mw:.2f} m_curl={m_curl:.2f} | d_RW={d_rw:.2f} r_curl={r_curl:.2f} | d_TW={d_tw:.2f} d_TI={d_ti:.2f} d_TM={d_tm:.2f} t_ext={t_ext}")

