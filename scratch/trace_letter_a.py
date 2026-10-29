import os
import sys

sys.path.insert(0, os.path.abspath("."))
site_packages = os.path.abspath(".venv/Lib/site-packages")
if os.path.exists(site_packages) and site_packages not in sys.path:
    sys.path.insert(0, site_packages)

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import json
from backend.vision.recognizer import VSLRecognizer
from backend.vision.alphabet_classifier import VSLAlphabetClassifier
from backend.vision.feature_extractor import extract_dual_hand_features

rec = VSLRecognizer()
clf = VSLAlphabetClassifier()

with open('backend/tests/fixtures/real_alphabet_landmarks.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

a_samples = [s for s in data['samples'] if s['letter_upper'] == 'A' and s.get('quality_status') == 'ok']

print(f"Tổng số mẫu chữ A thực tế: {len(a_samples)}")
print("=" * 80)
print(f"{'Frame':<6} | {'Phase':<12} | {'VSLAlphabetClassifier':<22} | {'recognizer.classify_geom':<25} | {'predict_landmarks (mode=letters)'}")
print("-" * 80)

for s in a_samples:
    lms = s['landmarks']
    frm = s['frame_idx']
    phase = s.get('phase', 'unknown')
    
    # 1. Kết quả từ VSLAlphabetClassifier mới (Task 3B.1)
    res_clf = clf.classify_detailed(lms)
    clf_str = f"{res_clf['letter']} ({res_clf['confidence']:.2f}, {res_clf['status']})"
    
    # 2. Kết quả từ hàm classify_geometry cũ trong recognizer.py
    _, meta = extract_dual_hand_features(lms, None)
    geom_sign, geom_conf = rec.classify_geometry(lms, meta['primary_states'])
    geom_str = f"{geom_sign} ({geom_conf:.2f})"
    
    # 3. Kết quả từ luồng camera thực tế: predict_landmarks
    pred_res = rec.predict_landmarks(lms, mode="letters")
    pred_str = f"sign='{pred_res['sign']}', label='{pred_res['label']}', conf={pred_res['confidence']}"
    
    print(f"{frm:<6} | {phase:<12} | {clf_str:<22} | {geom_str:<25} | {pred_str}")
    
    # Kiểm tra chi tiết trạng thái finger_states và MLP của recognizer
    thumb_curl = meta['primary_states'].get('thumb_curl', 0.5)
    d_ti = meta['primary_states'].get('thumb_index_dist', -1)
    print(f"       -> Details: thumb_curl={thumb_curl:.3f}, thumb_ext={meta['primary_states'].get('thumb_extended')}, reason_clf={res_clf['reason']}")
    print()

