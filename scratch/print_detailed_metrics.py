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

with open('backend/tests/fixtures/real_alphabet_landmarks.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

clf = VSLAlphabetClassifier()

letters = ['A', 'C', 'E', 'O', 'S', 'T', 'D', 'L', 'B', 'I', 'V', 'Y']

print(f"{'L':<2} | {'Frm':<3} | {'Phase':<11} | {'t_ext':<5} | {'i_ext':<5} | {'d_TI':<5} | {'d_TM':<5} | {'d_TW':<5} | {'d_IM':<5} | {'d_IW':<5} | {'d_MW':<5} | {'i_c':<4} | {'m_c':<4}")
print("-" * 90)

for s in data['samples']:
    if s.get('quality_status') != 'ok':
        continue
    l = s['letter_upper']
    if l not in letters:
        continue
    feat = clf.extract_features(s['landmarks'])
    if not feat:
        continue
    print(f"{l:<2} | {s['frame_idx']:<3} | {s.get('phase',''):<11} | {str(feat['thumb_ext']):<5} | {str(feat['index_ext']):<5} | {feat['d_TI']:<5.2f} | {feat['d_TM']:<5.2f} | {feat['d_TW']:<5.2f} | {feat['d_IM']:<5.2f} | {feat['d_IW']:<5.2f} | {feat['d_MW']:<5.2f} | {feat['index_curl']:<4.2f} | {feat['middle_curl']:<4.2f}")

