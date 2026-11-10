import sys, os
sys.path.insert(0, os.path.abspath("."))
import json
import numpy as np
from backend.vision.alphabet_classifier import AlphabetClassifier

npz = np.load('backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz', allow_pickle=True)
clf = AlphabetClassifier()

# User webcam frames
webcam_file = 'backend/data/diagnostics/pilot_pilot_20261106_2347.jsonl'
with open(webcam_file, 'r', encoding='utf-8') as f:
    webcam_lines = [json.loads(l.strip()) for l in f if l.strip()]

webcam_s = [
    line['geometric']['features'] 
    for line in webcam_lines 
    if line.get('expected_sign') == 'S' and 'features' in line.get('geometric', {})
]

# Let's check rule A in AlphabetClassifier!
# In AlphabetClassifier:
# Rule 8: Chữ A
# if not i_ext and not m_ext and not r_ext and not p_ext:
#     if d_tw >= A_D_TW_MIN and d_tm >= A_D_TM_MIN and d_iw <= A_D_IW_MAX and d_mw <= A_D_MW_MAX:
#         matches.append(("A", ...))
# What are A constants?
from backend.vision.alphabet_classifier import A_D_TW_MIN, A_D_TM_MIN, A_D_IW_MAX, A_D_MW_MAX
print(f"A constants: A_D_TW_MIN={A_D_TW_MIN}, A_D_TM_MIN={A_D_TM_MIN}, A_D_IW_MAX={A_D_IW_MAX}, A_D_MW_MAX={A_D_MW_MAX}")

# Did user webcam S frames match A?
a_matches = 0
for f in webcam_s:
    dtw = f.get('d_TW', 0)
    dtm = f.get('d_TM', 0)
    diw = f.get('d_IW', 0)
    dmw = f.get('d_MW', 0)
    if dtw >= A_D_TW_MIN and dtm >= A_D_TM_MIN and diw <= A_D_IW_MAX and dmw <= A_D_MW_MAX:
        a_matches += 1

print(f"Webcam S matched as A: {a_matches}/{len(webcam_s)}")

