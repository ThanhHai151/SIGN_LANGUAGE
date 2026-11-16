import os
import sys

sys.path.insert(0, os.path.abspath("."))
site_packages = os.path.abspath(".venv/Lib/site-packages")
if os.path.exists(site_packages) and site_packages not in sys.path:
    sys.path.insert(0, site_packages)

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import json
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

clf = VSLAlphabetClassifier()

with open('backend/tests/fixtures/real_alphabet_landmarks.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print("=== KIỂM TRA MẪU CỦA CHỮ U VÀ K ===")
for s in data['samples']:
    if s.get('quality_status') != 'ok':
        continue
    l = s['letter_upper']
    if l in ['U', 'K']:
        f = clf.extract_features(s['landmarks'])
        phase = s.get('phase', 'unknown')
        frm = s['frame_idx']
        print(f"[{l}] Frm {frm} ({phase}): i_ext={f['index_ext']} m_ext={f['middle_ext']} r_ext={f['ring_ext']} p_ext={f['pinky_ext']} | d_TI={f['d_TI']:.2f} d_TM={f['d_TM']:.2f} d_TW={f['d_TW']:.2f} d_IM={f['d_IM']:.2f} d_IW={f['d_IW']:.2f} d_MW={f['d_MW']:.2f} diff_x={f['diff_x_IM']:.2f}")

