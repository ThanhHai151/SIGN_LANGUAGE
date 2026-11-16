import os
import sys

sys.path.insert(0, os.path.abspath("."))
site_packages = os.path.abspath(".venv/Lib/site-packages")
if os.path.exists(site_packages) and site_packages not in sys.path:
    sys.path.insert(0, site_packages)

import json
from backend.vision.feature_extractor import extract_dual_hand_features

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

with open('backend/tests/fixtures/real_alphabet_landmarks.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for letter in ['S', 'T', 'C', 'O', 'D', 'L', 'A', 'E', 'B', 'I', 'Y', 'V']:
    samples = [s for s in data['samples'] if s['letter_upper'] == letter and s.get('quality_status') == 'ok']
    print(f"=== Letter {letter} ({len(samples)} samples) ===")
    for s in samples:
        _, meta = extract_dual_hand_features(s['landmarks'], None)
        st = meta.get('primary_states', {})
        t_ext = st.get('thumb_extended')
        i_ext = st.get('index_extended')
        m_ext = st.get('middle_extended')
        r_ext = st.get('ring_extended')
        p_ext = st.get('pinky_extended')
        i_curl = st.get('index_curl', 0.0)
        m_curl = st.get('middle_curl', 0.0)
        p_curl = st.get('pinky_curl', 0.0)
        phase = s.get('phase', 'unknown')
        print(f"  frame {s['frame_idx']} ({phase}): ext=[T:{t_ext} I:{i_ext} M:{m_ext} R:{r_ext} P:{p_ext}], curl=[I:{i_curl:.2f} M:{m_curl:.2f} P:{p_curl:.2f}]")
