import json
import sys
sys.path.insert(0, ".")

from backend.vision.recognizer import VSLRecognizer
from backend.vision.feature_extractor import normalize_hand_landmarks

rec = VSLRecognizer()

print("=== DIAGNOSING P (D0509N) ===")
with open("scratch/d0509n_extracted_raw.json", encoding="utf-8") as f:
    samples_p = json.load(f)

for idx, s in enumerate(samples_p):
    lm = s.get("landmarks")
    if not lm or len(lm) < 21:
        continue
    geom_res = rec.alphabet_classifier.classify_detailed(lm)
    feat_72, _ = normalize_hand_landmarks(lm)
    head_res = rec.static_alphabet_head.predict_feature(feat_72[:72], confidence_threshold=0.0) if rec.has_static_head else {}
    print(f"P frame {s['frame_index']:2d}: geom={geom_res.get('status')} {geom_res.get('letter')} conf={geom_res.get('confidence'):.2f} | head={head_res.get('raw_predicted_letter')} conf={head_res.get('confidence'):.2f} top3={head_res.get('top_k', [])[:3]}")

print("\n=== DIAGNOSING X (D0517N) ===")
with open("scratch/d0517n_extracted_raw.json", encoding="utf-8") as f:
    samples_x = json.load(f)

for idx, s in enumerate(samples_x):
    lm = s.get("landmarks")
    if not lm or len(lm) < 21:
        continue
    geom_res = rec.alphabet_classifier.classify_detailed(lm)
    feat_72, _ = normalize_hand_landmarks(lm)
    head_res = rec.static_alphabet_head.predict_feature(feat_72[:72], confidence_threshold=0.0) if rec.has_static_head else {}
    print(f"X frame {s['frame_index']:2d}: geom={geom_res.get('status')} {geom_res.get('letter')} conf={geom_res.get('confidence'):.2f} | head={head_res.get('raw_predicted_letter')} conf={head_res.get('confidence'):.2f} top3={head_res.get('top_k', [])[:3]}")

