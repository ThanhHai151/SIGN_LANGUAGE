import os
import sys
import json
sys.path.insert(0, os.path.abspath("."))
from backend.vision.alphabet_classifier import AlphabetClassifier
from scratch.task2_4_candidate_validation import eval_x_upright_custom, eval_x_forward, load_fixture

classifier = AlphabetClassifier()

print("--- Inspecting False Positives for Candidate 3 & 4 vs Candidate 2 ---")
for ltr in ['E', 'O', 'S']:
    fpath = f'backend/data/alphabet_v2/static_landmarks/{ltr}.json'
    d = load_fixture(fpath)
    for idx, s in enumerate(d.get('samples', [])):
        lm = s.get('landmarks')
        if not lm or len(lm) != 21:
            continue
        feat = classifier.extract_features(lm)
        if not feat:
            continue
        m2 = eval_x_upright_custom(feat, 0.25, 1.55, 1.30, 0.65)
        m3 = eval_x_upright_custom(feat, 0.30, 1.55, 1.30, 0.70)
        m4 = eval_x_upright_custom(feat, 0.30, 1.60, 1.35, 0.75)
        if m3 or m4:
            fidx = s.get('frame_index')
            print(f"Letter {ltr} idx={idx} frame_index={fidx}:")
            print(f"  m2={m2}, m3={m3}, m4={m4}")
            print(f"  d_TI={feat.get('d_TI'):.4f}, d_IW={feat.get('d_IW'):.4f}, d_TW={feat.get('d_TW'):.4f}, d_PW={feat.get('d_PW'):.4f}")
            print(f"  idx_straight={feat.get('idx_straight'):.4f}, index_curl={feat.get('index_curl'):.4f}, idx_tip_y={feat.get('idx_tip_y'):.4f}")
            # Also simulate pipeline behavior
            from scratch.task2_4_candidate_validation import simulate_pipeline
            sim3 = simulate_pipeline(classifier, lm, feat, 0.30, 1.55, 1.30, 0.70)
            print(f"  Simulate Pipeline Cand 3: status={sim3['status']}, letter={sim3['letter']}, matches={sim3['matches']}")

