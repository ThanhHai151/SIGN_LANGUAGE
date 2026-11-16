import sys
import os
import json
import math

sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.abspath("."))
from backend.vision.feature_extractor import extract_dual_hand_features, calculate_angle, calculate_distance
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

clf = VSLAlphabetClassifier()

motion_dir = 'backend/data/alphabet_v2/motion_sequences'
letters = ['Ă', 'Â', 'Ê', 'Ô', 'Ơ', 'Ư', 'Đ']

detailed_phases = {}

for l in letters:
    fpath = os.path.join(motion_dir, f"{l}.json")
    with open(fpath, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    seq = data['sequence']
    
    print(f"\n=======================================================")
    print(f"PHÂN TÍCH HÌNH THÁI VÀ PHA CHO CHỮ: {l}")
    print(f"=======================================================")
    
    frame_analysis = []
    for s in seq:
        if s['quality_status'] == 'ok':
            lms = s['landmarks']
            fidx = s['frame_index']
            ts = s['timestamp_sec']
            
            # 1. Thử phân loại bằng VSLAlphabetClassifier
            clf_res = clf.classify_detailed(lms)
            match_letter = clf_res.get('letter', 'UNKNOWN')
            status = clf_res.get('status')
            
            # 2. Trích xuất góc cổ tay và hướng lòng bàn tay
            # wrist=0, middle_mcp=9
            wrist = lms[0]
            middle_mcp = lms[9]
            # vector wrist -> middle_mcp biểu thị trục dọc bàn tay
            dy = middle_mcp[1] - wrist[1] # y tăng xuống
            dx = middle_mcp[0] - wrist[0]
            dz = middle_mcp[2] - wrist[2]
            
            # Ngón trỏ (8), ngón cái (4), ngón út (20)
            index_tip = lms[8]
            thumb_tip = lms[4]
            pinky_tip = lms[20]
            
            # Tính finger states
            _, meta = extract_dual_hand_features(lms, None)
            states = meta.get('primary_states', {})
            
            frame_analysis.append({
                'frame_idx': fidx,
                'ts': ts,
                'clf_match': match_letter,
                'clf_status': status,
                'wrist_y': round(wrist[1], 3),
                'thumb_ext': states.get('thumb_extended'),
                'index_ext': states.get('index_extended'),
                'middle_ext': states.get('middle_extended'),
                'ring_ext': states.get('ring_extended'),
                'pinky_ext': states.get('pinky_extended'),
                'thumb_curl': round(states.get('thumb_curl', 0), 2),
                'index_curl': round(states.get('index_curl', 0), 2)
            })
            
    # Sample and show progression
    print(f"Tổng số frame OK: {len(frame_analysis)}")
    print(f"{'Frame':<6} | {'Time':<6} | {'Clf Match':<10} | {'Status':<10} | {'T':<5} {'I':<5} {'M':<5} {'R':<5} {'P':<5} | {'Wrist Y'}")
    print("-" * 75)
    # Print every 3rd or 4th frame in active segment
    step = max(1, len(frame_analysis) // 12)
    for fa in frame_analysis[::step]:
        exts = f"{int(fa['thumb_ext'])} {int(fa['index_ext'])} {int(fa['middle_ext'])} {int(fa['ring_ext'])} {int(fa['pinky_ext'])}"
        print(f"{fa['frame_idx']:<6} | {fa['ts']:<6.2f} | {fa['clf_match']:<10} | {fa['clf_status']:<10} | {exts:<17} | {fa['wrist_y']}")
    
    detailed_phases[l] = frame_analysis

with open('backend/data/alphabet_v2/audit/detailed_phase_analysis.json', 'w', encoding='utf-8') as fp:
    json.dump(detailed_phases, fp, ensure_ascii=False, indent=2)

