import sys
sys.path.insert(0, '.')
import json
from backend.vision.alphabet_classifier import VSLAlphabetClassifier
from scratch.task1_3_1_eval_candidate import evaluate_p_candidate_conditions

clf = VSLAlphabetClassifier()

def inspect_frames(letter, target_frames):
    path = f'backend/data/alphabet_v2/static_landmarks/{letter}.json'
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f'=== INSPECTING ABNORMAL FRAMES IN {letter}.json ===')
    for s in data.get('samples', []):
        f_idx = s.get('frame_index') or s.get('frame_idx')
        if f_idx in target_frames:
            phase = s.get('phase')
            lms = s.get('landmarks')
            feat = clf.extract_features(lms)
            eval_res = evaluate_p_candidate_conditions(feat)
            print(f"\nFrame {f_idx} (phase: {phase}):")
            print(f"  idx_tip_y = {feat.get('idx_tip_y'):.3f}, idx_dy = {feat.get('idx_dy'):.3f}, mid_tip_y = {feat.get('mid_tip_y'):.3f}")
            print(f"  d_IW = {feat.get('d_IW'):.3f}, d_MW = {feat.get('d_MW'):.3f}, mid_straight = {feat.get('mid_straight'):.3f}")
            print(f"  d_TI = {feat.get('d_TI'):.3f}, d_TM = {feat.get('d_TM'):.3f}, d_TW = {feat.get('d_TW'):.3f}")
            print(f"  ang_IM_dir = {feat.get('ang_IM_dir'):.1f}, d_RW = {feat.get('d_RW'):.3f}, d_PW = {feat.get('d_PW'):.3f}")
            print(f"  Candidate P-down Matched? {eval_res['matched']}")
            print(f"  Failed Conditions: {eval_res['failed']}")

# K abnormal frames: 10, 46, 48
inspect_frames('K', [10, 46, 48])

# D abnormal frames: 14, 54, 56
inspect_frames('D', [14, 54, 56])

# Q abnormal frames: 42, 60, 62, 64
inspect_frames('Q', [42, 60, 62, 64])

