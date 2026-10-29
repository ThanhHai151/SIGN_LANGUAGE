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
from backend.vision.feature_extractor import calculate_distance, calculate_angle

clf = VSLAlphabetClassifier()

with open('backend/tests/fixtures/real_alphabet_landmarks.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

unsupported = ['G', 'H', 'K', 'M', 'N', 'P', 'Q', 'R', 'U', 'X']

for letter in unsupported:
    samples = [s for s in data['samples'] if s['letter_upper'] == letter and s.get('quality_status') == 'ok']
    print(f"\n=======================================================")
    print(f"LETTER {letter} ({len(samples)} samples ok)")
    print(f"=======================================================")
    for s in samples:
        f = clf.extract_features(s['landmarks'])
        phase = s.get('phase', 'unknown')
        frm = s['frame_idx']
        ext = f"[T:{int(f['thumb_ext'])} I:{int(f['index_ext'])} M:{int(f['middle_ext'])} R:{int(f['ring_ext'])} P:{int(f['pinky_ext'])}]"
        d_ti = f['d_TI']
        d_tm = f['d_TM']
        d_tw = f['d_TW']
        d_im = f['d_IM']
        d_iw = f['d_IW']
        d_mw = f['d_MW']
        d_pw = f['d_PW']
        diff_x = f['diff_x_IM']
        
        # Thêm tính hướng cổ tay - ngón trỏ hoặc ngón giữa (góc so với trục Y hoặc X)
        pts = s['landmarks']
        wrist = pts[0]
        index_mcp = pts[5]
        index_tip = pts[8]
        middle_tip = pts[12]
        thumb_tip = pts[4]
        
        # vector wrist -> index_mcp (hướng bàn tay)
        vec_hand_x = index_mcp[0] - wrist[0]
        vec_hand_y = index_mcp[1] - wrist[1]
        
        # vector index_mcp -> index_tip (hướng ngón trỏ)
        vec_idx_x = index_tip[0] - index_mcp[0]
        vec_idx_y = index_tip[1] - index_mcp[1]
        
        print(f"Frm {frm:<3} ({phase:<11}): ext={ext} | d_TI={d_ti:.2f} d_TM={d_tm:.2f} d_TW={d_tw:.2f} d_IM={d_im:.2f} | d_IW={d_iw:.2f} d_MW={d_mw:.2f} d_PW={d_pw:.2f} | diff_x={diff_x:.2f} | hand_dir=({vec_hand_x:.2f}, {vec_hand_y:.2f}) idx_dir=({vec_idx_x:.2f}, {vec_idx_y:.2f})")

